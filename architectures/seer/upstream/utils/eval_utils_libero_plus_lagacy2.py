import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'osmesa')
os.environ['PYOPENGL_PLATFORM'] = os.environ.get('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])

from pathlib import Path
import csv
import json
from datetime import datetime
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


def quaternion_to_euler(q):
    rot = R.from_quat(q)
    euler = rot.as_euler('xyz', degrees=False)
    return euler


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


def strip_known_extensions(name: str) -> str:
    name = str(name)
    if name.endswith(".pruned_init"):
        return name[:-12]
    if name.endswith(".init"):
        return name[:-5]
    if name.endswith(".bddl"):
        return name[:-5]
    return name


def resolve_render_gpu_device_id(local_rank: int) -> int:
    """
    robosuite EGL expects a device id that is actually listed in CUDA_VISIBLE_DEVICES.
    If CUDA_VISIBLE_DEVICES=4,5,6,7 and local_rank=0, we must pass 4, not 0.
    For osmesa, this value is effectively ignored, so this mapping is harmless there too.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible == "":
        return int(local_rank)

    visible_ids = [x.strip() for x in visible.split(",") if x.strip() != ""]
    if len(visible_ids) == 0:
        return int(local_rank)

    if local_rank < 0 or local_rank >= len(visible_ids):
        raise ValueError(
            f"local_rank={local_rank} is out of range for CUDA_VISIBLE_DEVICES={visible}"
        )

    return int(visible_ids[local_rank])


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
        mapping[str(suite_name)][strip_known_extensions(task_name)] = sorted(set(categories))

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
        strip_known_extensions(task.name),
        strip_known_extensions(task.bddl_file),
        strip_known_extensions(task.init_states_file),
    ]
    for candidate in candidates:
        if candidate in suite_map:
            return suite_map[candidate]
    return []


def build_bddl_index(libero_pkg_root: str, suite_name: str):
    bddl_dir = os.path.join(libero_pkg_root, "bddl_files", suite_name)
    if not os.path.isdir(bddl_dir):
        raise FileNotFoundError(f"Suite BDDL directory does not exist: {bddl_dir}")

    exact_index = {}
    stems = []
    for path in Path(bddl_dir).glob("*.bddl"):
        stem = path.stem
        exact_index[stem] = str(path)
        stems.append(stem)

    sorted_stems = sorted(stems, key=len, reverse=True)
    return bddl_dir, exact_index, sorted_stems


def build_init_index(libero_pkg_root: str, suite_name: str):
    init_dir = os.path.join(libero_pkg_root, "init_files", suite_name)
    if not os.path.isdir(init_dir):
        raise FileNotFoundError(f"Suite init directory does not exist: {init_dir}")

    stem_to_paths = defaultdict(list)
    for path in Path(init_dir).glob("*"):
        if not path.is_file():
            continue
        fname = path.name
        if fname.endswith(".pruned_init"):
            stem = fname[:-12]
        elif fname.endswith(".init"):
            stem = fname[:-5]
        else:
            continue
        stem_to_paths[stem].append(str(path))

    sorted_stems = sorted(stem_to_paths.keys(), key=len, reverse=True)
    return init_dir, stem_to_paths, sorted_stems


def choose_best_init_path(paths):
    pruned = sorted([p for p in paths if p.endswith(".pruned_init")])
    if pruned:
        return pruned[0]
    plain = sorted([p for p in paths if p.endswith(".init")])
    if plain:
        return plain[0]
    return sorted(paths)[0]


def resolve_by_existing_stems(candidates, sorted_stems):
    for cand in candidates:
        if not cand:
            continue
        for stem in sorted_stems:
            if cand == stem or cand.startswith(stem + "_"):
                return stem
    return None


def resolve_exact_bddl_path(libero_pkg_root: str, suite_name: str, task):
    bddl_dir, exact_index, sorted_stems = build_bddl_index(libero_pkg_root, suite_name)

    candidates = [
        strip_known_extensions(task.bddl_file),
        strip_known_extensions(task.name),
        strip_known_extensions(task.init_states_file),
    ]

    for cand in candidates:
        if cand in exact_index:
            return exact_index[cand], None

    matched_stem = resolve_by_existing_stems(candidates, sorted_stems)
    if matched_stem is not None:
        return exact_index[matched_stem], None

    available = sorted(list(exact_index.keys()))[:30]
    return None, (
        f"Could not resolve exact BDDL for suite={suite_name}, "
        f"task.name={task.name}, task.bddl_file={task.bddl_file}, task.init_states_file={task.init_states_file}. "
        f"Looked in: {bddl_dir}. First available BDDL stems: {available}"
    )


def resolve_exact_init_states_path(libero_pkg_root: str, suite_name: str, task):
    init_dir, stem_to_paths, sorted_stems = build_init_index(libero_pkg_root, suite_name)

    candidates = [
        strip_known_extensions(task.init_states_file),
        strip_known_extensions(task.name),
        strip_known_extensions(task.bddl_file),
    ]

    for cand in candidates:
        if cand in stem_to_paths:
            return choose_best_init_path(stem_to_paths[cand]), None

    matched_stem = resolve_by_existing_stems(candidates, sorted_stems)
    if matched_stem is not None:
        return choose_best_init_path(stem_to_paths[matched_stem]), None

    available = [os.path.basename(choose_best_init_path(paths)) for _, paths in sorted(stem_to_paths.items())[:30]]
    return None, (
        f"Could not resolve init file for suite={suite_name}, "
        f"task.name={task.name}, task.init_states_file={task.init_states_file}, task.bddl_file={task.bddl_file}. "
        f"Looked in: {init_dir}. First available init files: {available}"
    )


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


def save_eval_artifacts(suite_name, merged, task_level_results, category_summary, total_success):
    outdir = get_eval_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    missing_items = [x for x in merged if x["success"] is None]
    ok_items = [x for x in merged if x["success"] is not None]
    missing_task_ids = sorted(set(x["task_id"] for x in missing_items))
    valid_task_ids = sorted(set(x["task_id"] for x in task_level_results))
    total_task_ids = sorted(set(valid_task_ids) | set(missing_task_ids))

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
                "init_file": item.get("init_file"),
                "status": item.get("status"),
                "missing_reason": item.get("missing_reason"),
            }
        )

    write_csv(
        os.path.join(outdir, f"{suite_name}_raw_results_{timestamp}.csv"),
        fieldnames=[
            "suite", "task_id", "task_name", "success", "categories",
            "bddl", "init_file", "status", "missing_reason",
        ],
        rows=raw_rows,
    )

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
        fieldnames=["suite", "task_id", "task_name", "avg_success", "categories"],
        rows=task_rows,
    )

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
                "init_file": item.get("init_file"),
            }
        )

    write_csv(
        os.path.join(outdir, f"{suite_name}_missing_{timestamp}.csv"),
        fieldnames=[
            "suite", "task_id", "task_name", "categories", "status",
            "missing_reason", "bddl", "init_file",
        ],
        rows=missing_rows,
    )

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
            "suite", "category", "task_count_total", "task_count_valid",
            "task_count_missing", "avg_success",
        ],
        rows=category_rows,
    )

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
        "files": {
            "raw_results_csv": f"{suite_name}_raw_results_{timestamp}.csv",
            "task_summary_csv": f"{suite_name}_task_summary_{timestamp}.csv",
            "missing_csv": f"{suite_name}_missing_{timestamp}.csv",
            "category_summary_csv": f"{suite_name}_category_summary_{timestamp}.csv",
        },
    }

    write_json(os.path.join(outdir, f"{suite_name}_report_{timestamp}.json"), report)

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
                f"{category}: total={stat.get('task_count_total', 0)}, "
                f"valid={stat.get('task_count_valid', 0)}, "
                f"missing={stat.get('task_count_missing', 0)}, "
                f"avg_success={stat.get('avg_success', None)}\n"
            )
        f.write("\n[Missing Entries]\n")
        for item in missing_items:
            f.write(
                f"task_id={item.get('task_id')} task_name={item.get('task_name')} "
                f"categories={','.join(item.get('categories', []))} status={item.get('status')} "
                f"reason={item.get('missing_reason')}\n"
            )

    return {"outdir": outdir, "timestamp": timestamp, "txt_path": txt_path}


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
        image = obs["agentview_image"]
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image])
        image_x = image_x.unsqueeze(1).to(dtype=self.cast_type)

        gripper = obs["robot0_eye_in_hand_image"]
        gripper = Image.fromarray(gripper)
        gripper = self.image_process_fn([gripper])
        gripper = gripper.unsqueeze(1).to(dtype=self.cast_type)

        text_x = self.text_process_fn([goal])
        text_x = text_x.unsqueeze(1)
        state_pos = obs["robot0_eef_pos"]
        state_ori = quaternion_to_euler(obs["robot0_eef_quat"])

        if not self.gripper_width:
            state = torch.from_numpy(np.concatenate([state_pos, state_ori, self.gripper_state])).to(
                dtype=self.cast_type).unsqueeze(0).unsqueeze(0)
        else:
            state = torch.from_numpy(np.concatenate([state_pos, state_ori, obs['robot0_gripper_qpos']])).to(
                dtype=self.cast_type).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            text_x = text_x.to(device)
            gripper = gripper.to(device)
            state = state.to(device)

            self.img_queue.append(image_x)
            self.gripper_queue.append(gripper)
            self.state_queue.append(state)
            if len(self.text_queue) == 0 and text_x is not None:
                self.text_queue.append(text_x)
                for _ in range(self.model.module.sequence_length - 1):
                    self.text_queue.append(text_x)

            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)

            num_step = image_primary.shape[1]
            if num_step < self.history_len:
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

            if self.use_ensembling:
                if num_step < self.history_len:
                    selected_step = num_step - 1
                else:
                    selected_step = -1
                action = torch.concat((arm_action[:, selected_step], gripper_action[:, selected_step]), dim=-1)
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
                action[:, -1] = (action[:, -1] - 0.5) * 2
                action = action.detach().cpu().numpy()[-1]
            else:
                action = torch.concat((arm_action[:, -1], gripper_action[:, -1]), dim=-1)
                action = torch.concat((action[:, :6], action[:, 6:] > 0.5), dim=-1)
                action[:, -1] = (action[:, -1] - 0.5) * 2
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


def evaluate_policy_ddp(args, model):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.finetune_type]()
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", device_id))
    render_gpu_device_id = resolve_render_gpu_device_id(local_rank)

    libero_repo_root = args.libero_path
    libero_pkg_root = os.path.join(libero_repo_root, "libero", "libero")
    if not os.path.isdir(libero_pkg_root):
        raise FileNotFoundError(
            f"LIBERO package root does not exist: {libero_pkg_root} (args.libero_path={args.libero_path})"
        )

    classification_map = load_task_classification(libero_pkg_root)
    num_eval_episodes = 1 if getattr(args, "use_libero_plus", False) else 20
    task_num = len(task_suite.get_task_names())

    if device_id == 0:
        print(f"[LIBERO-PLUS EVAL] suite             : {args.finetune_type}")
        print(f"[LIBERO-PLUS EVAL] task_num          : {task_num}")
        print(f"[LIBERO-PLUS EVAL] num_eval_episodes : {num_eval_episodes}")
        print(f"[LIBERO-PLUS EVAL] local_rank        : {local_rank}")
        print(f"[LIBERO-PLUS EVAL] render_gpu_device : {render_gpu_device_id}")

    num_sequences = num_eval_episodes * task_num
    all_eval_sequences = list(range(num_sequences))
    split_eval_sequences = np.array_split(all_eval_sequences, device_num)
    local_eval_sequences = split_eval_sequences[device_id].tolist()
    eval_sequences = tqdm(local_eval_sequences, disable=(device_id != 0), desc=args.finetune_type)

    local_results = []

    local_success_count = 0
    local_failure_count = 0
    local_skip_count = 0
    local_processed_count = 0

    def print_progress(prefix: str):
        if device_id == 0:
            print(
                f"[LIBERO-PLUS EVAL][PROGRESS] {prefix} | "
                f"processed={local_processed_count}/{len(local_eval_sequences)} "
                f"success={local_success_count} failure={local_failure_count} skip={local_skip_count}"
            )

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
                    "init_file": None,
                    "status": "missing_bddl",
                    "missing_reason": missing_reason,
                }
            )

            local_skip_count += 1
            local_processed_count += 1
            if device_id == 0:
                print(f"[LIBERO-PLUS EVAL][SKIP] task_id={task_id} task_name={task_name} status=missing_bddl")
            print_progress(f"task_id={task_id} status=missing_bddl")
            continue

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": args.libero_img_size,
            "camera_widths": args.libero_img_size,
            "render_gpu_device_id": render_gpu_device_id,
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
                    "bddl": os.path.basename(task_bddl_file),
                    "init_file": None,
                    "status": "missing_init_states",
                    "missing_reason": missing_reason,
                }
            )
            local_skip_count += 1
            local_processed_count += 1
            env.close()
            if device_id == 0:
                print(f"[LIBERO-PLUS EVAL][SKIP] task_id={task_id} task_name={task_name} status=missing_init_states")
            print_progress(f"task_id={task_id} status=missing_init_states")
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
                    "bddl": os.path.basename(task_bddl_file),
                    "init_file": os.path.basename(init_states_path),
                    "status": "missing_init_state_index",
                    "missing_reason": missing_reason,
                }
            )
            local_skip_count += 1
            local_processed_count += 1
            env.close()
            if device_id == 0:
                print(
                    f"[LIBERO-PLUS EVAL][SKIP] task_id={task_id} task_name={task_name} status=missing_init_state_index")
            print_progress(f"task_id={task_id} status=missing_init_state_index")
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
                "init_file": os.path.basename(init_states_path),
                "status": "ok",
                "missing_reason": None,
            }
        )
        if int(result) == 1:
            local_success_count += 1
        else:
            local_failure_count += 1
        local_processed_count += 1
        print_progress(f"task_id={task_id} status=ok result={int(result)}")

    gathered = [None for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(local_results, gathered, dst=0)

    if torch.distributed.get_rank() != 0:
        return

    merged = []
    for chunk in gathered:
        merged.extend(chunk)

    global_success_count = sum(1 for x in merged if x["success"] == 1)
    global_failure_count = sum(1 for x in merged if x["success"] == 0)
    global_skip_count = sum(1 for x in merged if x["success"] is None)
    print(
        f"[LIBERO-PLUS EVAL] gathered results | "
        f"success={global_success_count} failure={global_failure_count} skip={global_skip_count} total={len(merged)}"
    )

    by_task = defaultdict(list)
    for item in merged:
        by_task[item["task_id"]].append(item)

    task_level_results = []
    for task_id in range(task_num):
        task_items = by_task.get(task_id, [])
        if len(task_items) == 0:
            continue

        valid_items = [x for x in task_items if x["success"] is not None]
        if len(valid_items) == 0:
            continue

        successes = np.array([x["success"] for x in valid_items], dtype=np.float32)
        avg_success = float(np.mean(successes)) if len(successes) > 0 else 0.0
        task = task_suite.get_task(task_id)
        categories = valid_items[0].get("categories", [])

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

    missing_items = [x for x in merged if x["success"] is None]
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

    print(f"[LIBERO-PLUS EVAL] Saved reports to: {artifact_info['outdir']}")

    header = ["Suite"] + CATEGORY_NAMES + ["Total"]
    row = [args.finetune_type]
    for category in CATEGORY_NAMES:
        value = category_summary[category]["avg_success"]
        row.append("-" if value is None else f"{value * 100:.1f}")
    row.append("-" if total_success is None else f"{total_success * 100:.1f}")

    print("\nBenchmark summary")
    print("\t".join(header))
    print("\t".join(row))


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
        gripper_width=args.gripper_width)
    evaluate_policy_ddp(args, wrapped_model)
