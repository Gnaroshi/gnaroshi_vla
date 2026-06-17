import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"

import json
import re
import csv
from datetime import datetime
from collections import Counter
from pathlib import Path
from collections import deque, defaultdict
import functools

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
from PIL import Image

from utils.data_utils import preprocess_image, preprocess_text_calvin
from utils.train_utils import get_cast_dtype

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

CATEGORY_NAMES = [
    "Camera",
    "Robot",
    "Language",
    "Light",
    "Background",
    "Noise",
    "Layout",
]

CATEGORY_ALIASES = {
    "camera": "Camera",
    "camera viewpoints": "Camera",
    "view": "Camera",
    "robot": "Robot",
    "robot initial states": "Robot",
    "language": "Language",
    "language instructions": "Language",
    "light": "Light",
    "light conditions": "Light",
    "background": "Background",
    "background textures": "Background",
    "noise": "Noise",
    "sensor noise": "Noise",
    "layout": "Layout",
    "objects layout": "Layout",
}

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@ MUJOCO_GL         :", os.environ.get("MUJOCO_GL"))


def get_eval_output_dir():
    outdir = os.environ.get("LIBERO_PLUS_EVAL_OUTDIR", None)
    if outdir is None or outdir.strip() == "":
        outdir = os.path.join(os.getcwd(), "eval_libero_plus_outputs")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def safe_float(x):
    if x is None:
        return None
    return float(x)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_eval_artifacts(
        suite_name,
        merged,
        task_level_results,
        category_summary,
        total_success,
):
    outdir = get_eval_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    missing_items = [x for x in merged if x["success"] is None]
    ok_items = [x for x in merged if x["success"] is not None]

    missing_task_ids = sorted(set(x["task_id"] for x in missing_items))
    valid_task_ids = sorted(set(x["task_id"] for x in task_level_results))
    total_task_ids = sorted(set(valid_task_ids) | set(missing_task_ids))

    # 1) raw per-trial results
    raw_rows = []
    for item in merged:
        raw_rows.append(
            {
                "suite": item.get("suite"),
                "task_id": item.get("task_id"),
                "task_name": item.get("task_name"),
                "success": item.get("success"),
                "categories": ",".join(item.get("categories", [])),
                "bddl": item.get("bddl"),
                "status": item.get("status"),
                "missing_reason": item.get("missing_reason"),
            }
        )

    write_csv(
        os.path.join(outdir, f"{suite_name}_raw_results_{timestamp}.csv"),
        fieldnames=[
            "suite",
            "task_id",
            "task_name",
            "success",
            "categories",
            "bddl",
            "status",
            "missing_reason",
        ],
        rows=raw_rows,
    )

    # 2) per-task summary
    task_rows = []
    for item in task_level_results:
        task_rows.append(
            {
                "suite": suite_name,
                "task_id": item["task_id"],
                "task_name": item["task_name"],
                "avg_success": safe_float(item["avg_success"]),
                "categories": ",".join(item.get("categories", [])),
            }
        )

    write_csv(
        os.path.join(outdir, f"{suite_name}_task_summary_{timestamp}.csv"),
        fieldnames=[
            "suite",
            "task_id",
            "task_name",
            "avg_success",
            "categories",
        ],
        rows=task_rows,
    )

    # 3) missing/skipped only
    missing_rows = []
    for item in missing_items:
        missing_rows.append(
            {
                "suite": item.get("suite"),
                "task_id": item.get("task_id"),
                "task_name": item.get("task_name"),
                "categories": ",".join(item.get("categories", [])),
                "status": item.get("status"),
                "missing_reason": item.get("missing_reason"),
                "bddl": item.get("bddl"),
            }
        )

    write_csv(
        os.path.join(outdir, f"{suite_name}_missing_{timestamp}.csv"),
        fieldnames=[
            "suite",
            "task_id",
            "task_name",
            "categories",
            "status",
            "missing_reason",
            "bddl",
        ],
        rows=missing_rows,
    )

    # 4) category aggregate
    category_rows = []
    for category in CATEGORY_NAMES:
        stat = category_summary.get(category, {})
        category_rows.append(
            {
                "suite": suite_name,
                "category": category,
                "task_count_total": stat.get("task_count_total", 0),
                "task_count_valid": stat.get("task_count_valid", 0),
                "task_count_missing": stat.get("task_count_missing", 0),
                "avg_success": safe_float(stat.get("avg_success")),
            }
        )

    # Total row
    category_rows.append(
        {
            "suite": suite_name,
            "category": "Total",
            "task_count_total": len(total_task_ids),
            "task_count_valid": len(valid_task_ids),
            "task_count_missing": len(missing_task_ids),
            "avg_success": safe_float(total_success),
        }
    )

    write_csv(
        os.path.join(outdir, f"{suite_name}_category_summary_{timestamp}.csv"),
        fieldnames=[
            "suite",
            "category",
            "task_count_total",
            "task_count_valid",
            "task_count_missing",
            "avg_success",
        ],
        rows=category_rows,
    )

    # 5) compact json report
    report = {
        "suite": suite_name,
        "timestamp": timestamp,
        "num_trials_total": len(merged),
        "num_trials_valid": len(ok_items),
        "num_trials_missing": len(missing_items),
        "num_task_total": len(total_task_ids),
        "num_task_valid": len(valid_task_ids),
        "num_task_missing": len(missing_task_ids),
        "category_summary": category_summary,
        "total_success": total_success,
        "missing_status_counter": dict(Counter([x.get("status") for x in missing_items])),
        "files": {
            "raw_results_csv": f"{suite_name}_raw_results_{timestamp}.csv",
            "task_summary_csv": f"{suite_name}_task_summary_{timestamp}.csv",
            "missing_csv": f"{suite_name}_missing_{timestamp}.csv",
            "category_summary_csv": f"{suite_name}_category_summary_{timestamp}.csv",
        },
    }

    write_json(
        os.path.join(outdir, f"{suite_name}_report_{timestamp}.json"),
        report,
    )

    # 6) human-readable txt
    txt_path = os.path.join(outdir, f"{suite_name}_report_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Suite: {suite_name}\n")
        f.write(f"Total trials: {len(merged)}\n")
        f.write(f"Valid trials: {len(ok_items)}\n")
        f.write(f"Missing trials: {len(missing_items)}\n")
        f.write(f"Total tasks: {len(total_task_ids)}\n")
        f.write(f"Valid tasks: {len(valid_task_ids)}\n")
        f.write(f"Missing tasks: {len(missing_task_ids)}\n")
        f.write(f"Total success: {total_success if total_success is not None else 'None'}\n")
        f.write("\n[Category Summary]\n")
        for category in CATEGORY_NAMES:
            stat = category_summary.get(category, {})
            f.write(
                f"{category}: "
                f"total={stat.get('task_count_total', 0)}, "
                f"valid={stat.get('task_count_valid', 0)}, "
                f"missing={stat.get('task_count_missing', 0)}, "
                f"avg_success={stat.get('avg_success', None)}\n"
            )

        f.write("\n[Missing Entries]\n")
        for item in missing_items:
            f.write(
                f"task_id={item.get('task_id')} "
                f"task_name={item.get('task_name')} "
                f"categories={','.join(item.get('categories', []))} "
                f"status={item.get('status')} "
                f"reason={item.get('missing_reason')}\n"
            )

    return {
        "outdir": outdir,
        "timestamp": timestamp,
        "txt_path": txt_path,
    }


def quaternion_to_euler(q):
    rot = R.from_quat(q)
    euler = rot.as_euler("xyz", degrees=False)
    return euler


def normalize_name(name: str) -> str:
    name = str(name)
    if name.endswith(".bddl"):
        name = name[:-5]
    return name


def strip_init_variant_suffixes(name: str) -> str:
    """
    init file 매핑용 suffix 제거.
    init은 init_files 디렉토리 기준으로만 찾는다.
    """
    name = str(name)
    if name.endswith(".pruned_init"):
        name = name[:-12]
    elif name.endswith(".init"):
        name = name[:-5]
    elif name.endswith(".bddl"):
        name = name[:-5]

    patterns = [
        r"_initstate_\d+$",
        r"_view_-?\d+(?:_-?\d+)*$",
        r"_table_\d+$",
        r"_tb_\d+$",
        r"_light_\d+$",
        r"_language_\d+$",
        r"_background_\d+$",
        r"_noise_\d+$",
        r"_robot_\d+$",
        r"_layout_\d+$",
        r"_camera_\d+$",
        r"_add_\d+$",
        r"_level\d+_sample\d+$",
        r"_sample\d+$",
    ]

    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            stripped = re.sub(pattern, "", name)
            if stripped != name:
                name = stripped
                changed = True

    return name


def parse_category_value(value):
    categories = set()
    if value is None:
        return []

    if isinstance(value, str):
        normalized = CATEGORY_ALIASES.get(value.strip().lower())
        if normalized is not None:
            categories.add(normalized)
    elif isinstance(value, list):
        for item in value:
            categories.update(parse_category_value(item))
    elif isinstance(value, dict):
        for key, inner in value.items():
            key_norm = CATEGORY_ALIASES.get(str(key).strip().lower())
            if isinstance(inner, bool):
                if inner and key_norm is not None:
                    categories.add(key_norm)
            else:
                categories.update(parse_category_value(inner))
                if key_norm is not None and inner:
                    categories.add(key_norm)

    return sorted(categories)


def load_task_classification(libero_pkg_root):
    classification_path = os.path.join(libero_pkg_root, "benchmark", "task_classification.json")
    mapping = defaultdict(dict)

    if not os.path.exists(classification_path):
        raise FileNotFoundError(f"task_classification.json not found: {classification_path}")

    with open(classification_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def add_entry(suite_name, task_name, categories):
        if not suite_name or not task_name:
            return
        mapping[str(suite_name)][normalize_name(task_name)] = sorted(set(categories))

    def parse_suite_item(suite_name, item):
        if not isinstance(item, dict):
            return
        task_name = item.get("name") or item.get("task_name") or item.get("bddl")
        categories = parse_category_value(item.get("category") or item.get("categories"))
        add_entry(suite_name, task_name, categories)

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            suite_name = item.get("suite") or item.get("suite_name") or item.get("libero_suite")
            parse_suite_item(suite_name, item)

    elif isinstance(raw, dict):
        for key, value in raw.items():
            suite_name = key

            if isinstance(value, list):
                for item in value:
                    parse_suite_item(suite_name, item)

            elif isinstance(value, dict):
                if ("name" in value) or ("task_name" in value) or ("bddl" in value):
                    parse_suite_item(suite_name, value)
                else:
                    for task_name, category_value in value.items():
                        categories = parse_category_value(category_value)
                        add_entry(suite_name, task_name, categories)

            else:
                raise ValueError(
                    f"Unsupported value type inside task_classification.json for key {suite_name!r}: {type(value)}"
                )
    else:
        raise ValueError(f"Unsupported task_classification.json format: {type(raw)}")

    print(f"[LIBERO-PLUS EVAL] loaded task classification: {classification_path}")
    for suite_name, suite_map in mapping.items():
        print(f"[LIBERO-PLUS EVAL] classification entries for {suite_name}: {len(suite_map)}")

    return mapping


def resolve_task_categories(suite_name, task, classification_map):
    suite_map = classification_map.get(suite_name, {})

    candidates = [
        normalize_name(task.name),
        normalize_name(task.bddl_file),
        strip_init_variant_suffixes(task.name),
        strip_init_variant_suffixes(task.bddl_file),
    ]

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in suite_map:
            return suite_map[candidate]

    return []


def build_suite_bddl_index(libero_pkg_root: str, suite_name: str):
    bddl_dir = os.path.join(libero_pkg_root, "bddl_files", suite_name)
    if not os.path.isdir(bddl_dir):
        raise FileNotFoundError(f"Suite BDDL directory does not exist: {bddl_dir}")

    exact_index = {}
    stripped_index = defaultdict(list)

    for path in Path(bddl_dir).glob("*.bddl"):
        stem = path.stem
        exact_index[stem] = str(path)
        stripped = strip_init_variant_suffixes(stem)
        stripped_index[stripped].append(str(path))

    return bddl_dir, exact_index, stripped_index


def resolve_exact_bddl_path(libero_pkg_root: str, suite_name: str, task):
    """
    성공 시: (path, None)
    실패 시: (None, reason)
    """
    bddl_dir, exact_index, stripped_index = build_suite_bddl_index(libero_pkg_root, suite_name)

    candidates = [
        normalize_name(task.bddl_file),
        normalize_name(task.name),
        strip_init_variant_suffixes(task.bddl_file),
        strip_init_variant_suffixes(task.name),
    ]

    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            ordered.append(c)
            seen.add(c)

    for c in ordered:
        if c in exact_index:
            return exact_index[c], None

    matched_paths = []
    for c in ordered:
        matched_paths.extend(stripped_index.get(c, []))

    matched_paths = sorted(set(matched_paths))

    if len(matched_paths) == 1:
        return matched_paths[0], None

    if len(matched_paths) > 1:
        reason = (
            f"Ambiguous BDDL mapping for suite={suite_name}, "
            f"task.name={task.name}, task.bddl_file={task.bddl_file}. "
            f"Matched files: {matched_paths}"
        )
        return None, reason

    available = sorted(exact_index.keys())[:30]
    reason = (
        f"Could not resolve exact BDDL for suite={suite_name}, "
        f"task.name={task.name}, task.bddl_file={task.bddl_file}. "
        f"Looked in: {bddl_dir}. "
        f"First available BDDL stems: {available}"
    )
    return None, reason


def build_suite_init_index(libero_pkg_root: str, suite_name: str):
    init_dir = os.path.join(libero_pkg_root, "init_files", suite_name)
    if not os.path.isdir(init_dir):
        raise FileNotFoundError(f"Suite init directory does not exist: {init_dir}")

    exact_index = {}
    stripped_index = defaultdict(list)

    for path in Path(init_dir).glob("*"):
        if not path.is_file():
            continue
        if path.suffix not in [".init", ".pruned_init"]:
            continue

        stem = path.name
        if stem.endswith(".pruned_init"):
            stem_no_ext = stem[:-12]
        elif stem.endswith(".init"):
            stem_no_ext = stem[:-5]
        else:
            continue

        exact_index[stem_no_ext] = exact_index.get(stem_no_ext, [])
        exact_index[stem_no_ext].append(str(path))

        stripped = strip_init_variant_suffixes(stem_no_ext)
        stripped_index[stripped].append(str(path))

    return init_dir, exact_index, stripped_index


def choose_best_init_path(paths):
    pruned = sorted([p for p in paths if p.endswith(".pruned_init")])
    if len(pruned) > 0:
        return pruned[0]

    plain = sorted([p for p in paths if p.endswith(".init")])
    if len(plain) > 0:
        return plain[0]

    return sorted(paths)[0]


def resolve_exact_init_states_path(libero_pkg_root: str, suite_name: str, task):
    """
    init은 init_files/<suite>/ 실제 파일 목록만 기준으로 찾는다.
    bddl 파일명은 사용하지 않는다.
    성공 시: (path, None)
    실패 시: (None, reason)
    """
    init_dir, exact_index, stripped_index = build_suite_init_index(libero_pkg_root, suite_name)

    raw_init_name = str(task.init_states_file)
    raw_task_name = str(task.name)

    if raw_init_name.endswith(".pruned_init"):
        init_stem = raw_init_name[:-12]
    elif raw_init_name.endswith(".init"):
        init_stem = raw_init_name[:-5]
    else:
        init_stem = raw_init_name

    candidates = [
        init_stem,
        raw_task_name,
        strip_init_variant_suffixes(init_stem),
        strip_init_variant_suffixes(raw_task_name),
    ]

    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            ordered.append(c)
            seen.add(c)

    # 1) exact stem match
    for c in ordered:
        if c in exact_index:
            return choose_best_init_path(exact_index[c]), None

    # 2) stripped stem match
    matched_paths = []
    for c in ordered:
        matched_paths.extend(stripped_index.get(c, []))

    matched_paths = sorted(set(matched_paths))
    if len(matched_paths) > 0:
        return choose_best_init_path(matched_paths), None

    available = []
    for stem, paths in exact_index.items():
        available.append(os.path.basename(choose_best_init_path(paths)))
    available = sorted(available)[:30]

    reason = (
        f"Could not resolve init file for suite={suite_name}, "
        f"task.name={task.name}, task.init_states_file={task.init_states_file}. "
        f"Looked in: {init_dir}. "
        f"First available init files: {available}"
    )
    return None, reason


class ModelWrapper:
    def __init__(self, model, tokenizer, image_processor, cast_dtype, history_len=10,
                 use_ensembling=False, ensembling_temp=0.01, libero_eval_max_steps=600, action_pred_steps=3,
                 gripper_width=False):
        super().__init__()
        self.model = model
        self.cast_type = cast_dtype
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.history_len = history_len
        self.libero_eval_max_steps = libero_eval_max_steps
        self.action_pred_steps = action_pred_steps
        self.device = "cuda"
        self.use_ensembling = use_ensembling
        self.ensembling_temp = ensembling_temp
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.act_queue = deque(maxlen=history_len - 1)
        self.cnt = 0
        self.gripper_width = gripper_width
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                [
                    self.libero_eval_max_steps,
                    self.libero_eval_max_steps + self.action_pred_steps,
                    7,
                ]
            ).to(self.device)

    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.act_queue = deque(maxlen=self.history_len - 1)
        self.gripper_state = np.array([-1.0])
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                [
                    self.libero_eval_max_steps,
                    self.libero_eval_max_steps + self.action_pred_steps,
                    7,
                ]
            ).to(self.device)

        self.cnt += 1

    def step(self, obs, goal, timestep):
        # preprocess image
        image = obs["agentview_image"]
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image])
        # expand image dimension
        image_x = image_x.unsqueeze(1).to(dtype=self.cast_type)

        gripper = obs["robot0_eye_in_hand_image"]
        gripper = Image.fromarray(gripper)
        gripper = self.image_process_fn([gripper])
        # expand image dimension
        gripper = gripper.unsqueeze(1).to(dtype=self.cast_type)

        # expand text dimension
        text_x = self.text_process_fn([goal])
        text_x = text_x.unsqueeze(1)
        state_pos = obs["robot0_eef_pos"]
        state_ori = quaternion_to_euler(obs["robot0_eef_quat"])

        if not self.gripper_width:
            state = torch.from_numpy(np.concatenate([state_pos, state_ori, self.gripper_state])).to(
                dtype=self.cast_type).unsqueeze(0).unsqueeze(0)  # [1, 1, 7]
        else:
            state = torch.from_numpy(np.concatenate([state_pos, state_ori, obs['robot0_gripper_qpos']])).to(
                dtype=self.cast_type).unsqueeze(0).unsqueeze(0)  # [1, 1, 8]

        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            text_x = text_x.to(device)
            gripper = gripper.to(device)
            state = state.to(device)

            self.img_queue.append(
                image_x)  # TODO find out how the policy completes the 5 sub-tasks. the obs of the later task will be appended after the former?
            self.gripper_queue.append(gripper)
            self.state_queue.append(state)
            if len(self.text_queue) == 0 and text_x is not None:  # the instruction does not change
                self.text_queue.append(text_x)
                for _ in range(self.model.module.sequence_length - 1):
                    self.text_queue.append(text_x)

            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)

            num_step = image_primary.shape[1]
            if num_step < self.history_len:  # padding
                input_image_primary = torch.cat(
                    [image_primary, image_primary[:, -1].repeat(1, self.history_len - num_step, 1, 1, 1)], dim=1)
                input_image_wrist = torch.cat(
                    [image_wrist, image_wrist[:, -1].repeat(1, self.history_len - num_step, 1, 1, 1)], dim=1)
                input_state = torch.cat([state, state[:, -1].repeat(1, self.history_len - num_step, 1)], dim=1)
            else:
                input_image_primary = image_primary
                input_image_wrist = image_wrist
                input_state = state

            arm_action, gripper_action, _, _, _, _ = self.model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=torch.zeros(1, self.history_len, 7).to(input_state.device),
            )

            # This need to align libero environment
            if self.use_ensembling:
                if num_step < self.history_len:
                    selected_step = num_step - 1
                else:
                    selected_step = -1
                action = torch.concat((arm_action[:, selected_step], gripper_action[:, selected_step]),
                                      dim=-1)  # (1, action_pred_steps, 7)
                self.all_time_actions[timestep:timestep + 1, timestep:timestep + self.action_pred_steps] = action
                actions_for_curr_step = self.all_time_actions[:, timestep]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = self.ensembling_temp
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
                action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                action = torch.concat((action[:, :6], action[:, 6:] > 0.5), dim=-1)
                action[:, -1] = (action[:, -1] - 0.5) * 2  # scale to -1 or 1
                action = action.detach().cpu().numpy()[-1]

        self.gripper_state = np.array([action[-1]])
        return action


def evaluate_libero_task(task, env, obs, args, model):
    steps = 0
    success = 0
    model.reset()
    goal = task.language
    with torch.no_grad():
        while steps < args.libero_eval_max_steps:
            action = model.step(obs, goal, steps)
            steps += 1
            obs, reward, done, info = env.step(action)
            if done:
                success = 1
                break
    env.close()
    return success


def get_num_eval_episodes(args):
    return 1 if getattr(args, "use_libero_plus", False) else 20


def evaluate_policy_ddp(args, model):
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.finetune_type not in benchmark_dict:
        raise KeyError(f"Unsupported suite key: {args.finetune_type}. Available keys: {list(benchmark_dict.keys())}")

    task_suite = benchmark_dict[args.finetune_type]()
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()

    libero_repo_root = args.libero_path
    libero_pkg_root = os.path.join(libero_repo_root, "libero", "libero")
    if not os.path.isdir(libero_pkg_root):
        raise FileNotFoundError(
            f"LIBERO package root does not exist: {libero_pkg_root} (args.libero_path={args.libero_path})"
        )

    classification_map = load_task_classification(libero_pkg_root)
    num_eval_episodes = get_num_eval_episodes(args)
    task_num = len(task_suite.get_task_names())

    if device_id == 0:
        print(f"[LIBERO-PLUS EVAL] suite             : {args.finetune_type}")
        print(f"[LIBERO-PLUS EVAL] args.libero_path  : {args.libero_path}")
        print(f"[LIBERO-PLUS EVAL] libero_pkg_root   : {libero_pkg_root}")
        print(f"[LIBERO-PLUS EVAL] task_num          : {task_num}")
        print(f"[LIBERO-PLUS EVAL] num_eval_episodes : {num_eval_episodes}")

    num_sequences = num_eval_episodes * task_num
    all_eval_sequences = list(range(num_sequences))
    split_eval_sequences = np.array_split(all_eval_sequences, device_num)
    local_eval_sequences = split_eval_sequences[device_id].tolist()
    eval_sequences = tqdm(local_eval_sequences, disable=(device_id != 0), desc=args.finetune_type)

    local_results = []
    for eval_id in eval_sequences:
        task_id = eval_id // num_eval_episodes
        exp_id = eval_id % num_eval_episodes

        task = task_suite.get_task(task_id)
        task_name = task.name
        task_categories = resolve_task_categories(args.finetune_type, task, classification_map)

        task_bddl_file, missing_reason = resolve_exact_bddl_path(
            libero_pkg_root=libero_pkg_root,
            suite_name=args.finetune_type,
            task=task,
        )

        if task_bddl_file is None:
            local_results.append(
                {
                    "suite": args.finetune_type,
                    "task_id": task_id,
                    "task_name": task_name,
                    "success": None,
                    "categories": task_categories,
                    "bddl": None,
                    "status": "missing_bddl",
                    "missing_reason": missing_reason,
                }
            )
            continue

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": args.libero_img_size,
            "camera_widths": args.libero_img_size,
            "render_gpu_device_id": device_id,
        }
        env = OffScreenRenderEnv(**env_args)
        env.task_id = task_id
        env.task_name = task_name
        env.task_suite_name = args.finetune_type
        env.reset()
        env.seed(args.seed)

        init_states_path, missing_reason = resolve_exact_init_states_path(
            libero_pkg_root=libero_pkg_root,
            suite_name=args.finetune_type,
            task=task,
        )

        if init_states_path is None:
            local_results.append(
                {
                    "suite": args.finetune_type,
                    "task_id": task_id,
                    "task_name": task_name,
                    "success": None,
                    "categories": task_categories,
                    "bddl": os.path.basename(task_bddl_file) if task_bddl_file else None,
                    "status": "missing_init_states",
                    "missing_reason": missing_reason,
                }
            )
            env.close()
            if device_id == 0:
                print(
                    f"[LIBERO-PLUS EVAL][SKIP] "
                    f"task_id={task_id} task_name={task_name} status=missing_init_states"
                )
            continue

        init_states = torch.load(init_states_path)

        if exp_id >= len(init_states):
            missing_reason = (
                f"Requested init state index {exp_id} out of range for {init_states_path}. "
                f"num_init_states={len(init_states)}"
            )
            local_results.append(
                {
                    "suite": args.finetune_type,
                    "task_id": task_id,
                    "task_name": task_name,
                    "success": None,
                    "categories": task_categories,
                    "bddl": os.path.basename(task_bddl_file) if task_bddl_file else None,
                    "status": "missing_init_state_index",
                    "missing_reason": missing_reason,
                }
            )
            env.close()
            if device_id == 0:
                print(
                    f"[LIBERO-PLUS EVAL][SKIP] "
                    f"task_id={task_id} task_name={task_name} status=missing_init_state_index"
                )
            continue

        init_state = init_states[exp_id]
        obs = env.set_init_state(init_state)

        for _ in range(5):
            env.step(np.zeros(7))

        result = evaluate_libero_task(task, env, obs, args, model)
        local_results.append(
            {
                "suite": args.finetune_type,
                "task_id": task_id,
                "task_name": task_name,
                "success": int(result),
                "categories": task_categories,
                "bddl": os.path.basename(task_bddl_file),
                "status": "ok",
                "missing_reason": None,
            }
        )

    gathered = [None for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(local_results, gathered, dst=0)

    if torch.distributed.get_rank() != 0:
        return

    merged = []
    for chunk in gathered:
        merged.extend(chunk)

    missing_items = [x for x in merged if x["success"] is None]

    by_task = defaultdict(list)
    for item in merged:
        by_task[item["task_id"]].append(item)

    task_level_results = []
    missing_task_counter_by_category = {name: 0 for name in CATEGORY_NAMES}

    for task_id in range(task_num):
        task_items = by_task.get(task_id, [])
        if len(task_items) == 0:
            continue

        valid_items = [x for x in task_items if x["success"] is not None]
        task = task_suite.get_task(task_id)

        categories = []
        if len(task_items) > 0:
            categories = task_items[0].get("categories", [])

        if len(valid_items) == 0:
            for category in categories:
                if category in missing_task_counter_by_category:
                    missing_task_counter_by_category[category] += 1
            continue

        successes = np.array([x["success"] for x in valid_items], dtype=np.float32)
        avg_success = float(np.mean(successes)) if len(successes) > 0 else 0.0

        task_level_results.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "avg_success": avg_success,
                "categories": categories,
            }
        )

    category_success = {name: [] for name in CATEGORY_NAMES}
    category_task_total = {name: 0 for name in CATEGORY_NAMES}

    for item in task_level_results:
        for category in item["categories"]:
            if category in category_success:
                category_success[category].append(item["avg_success"])
                category_task_total[category] += 1

    for item in missing_items:
        for category in item.get("categories", []):
            if category in category_task_total:
                category_task_total[category] += 1

    total_success = float(np.mean([x["avg_success"] for x in task_level_results])) if len(
        task_level_results) > 0 else None

    category_summary = {}
    for category in CATEGORY_NAMES:
        vals = category_success[category]
        category_summary[category] = {
            "task_count_total": category_task_total[category],
            "task_count_valid": len(vals),
            "task_count_missing": category_task_total[category] - len(vals),
            "avg_success": float(np.mean(vals)) if len(vals) > 0 else None,
        }

    artifact_info = save_eval_artifacts(
        suite_name=args.finetune_type,
        merged=merged,
        task_level_results=task_level_results,
        category_summary=category_summary,
        total_success=total_success,
    )

    print(f"\n[LIBERO-PLUS EVAL] Saved reports to: {artifact_info['outdir']}")

    header = ["Suite"] + CATEGORY_NAMES + ["Total"]
    row = [args.finetune_type]
    for category in CATEGORY_NAMES:
        value = category_summary[category]["avg_success"]
        row.append("-" if value is None else f"{value * 100:.1f}")
    row.append("-" if total_success is None else f"{total_success * 100:.1f}")

    print("\nBenchmark summary")
    print("\t".join(header))
    print("\t".join(row))

    print("\n[LIBERO-PLUS EVAL] Missing counts by category")
    for category in CATEGORY_NAMES:
        print(
            f"{category}: total={category_summary[category]['task_count_total']} "
            f"valid={category_summary[category]['task_count_valid']} "
            f"missing={category_summary[category]['task_count_missing']}"
        )


def eval_one_epoch_libero_ddp(args, model, image_processor, tokenizer):
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = args.sequence_length
    wrapped_model = ModelWrapper(
        model,
        tokenizer,
        image_processor,
        cast_dtype,
        history_len=hist_len,
        use_ensembling=args.eval_libero_ensembling,
        ensembling_temp=args.ensembling_temp,
        libero_eval_max_steps=args.libero_eval_max_steps,
        action_pred_steps=args.action_pred_steps,
        gripper_width=args.gripper_width,
    )
    evaluate_policy_ddp(args, wrapped_model)
