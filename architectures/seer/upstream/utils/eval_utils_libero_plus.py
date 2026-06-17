import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'osmesa')
os.environ['PYOPENGL_PLATFORM'] = os.environ.get('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])

from pathlib import Path
import copy
import csv
import json
import numpy as np
import torch
from collections import deque, defaultdict
import functools
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

from utils.data_utils import preprocess_image, preprocess_text_calvin
from utils.train_utils import get_cast_dtype

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image


def quaternion_to_euler(q):
    rot = R.from_quat(q)
    euler = rot.as_euler('xyz', degrees=False)
    return euler


benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

CATEGORY_ORDER = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
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

SUPPORTED_SUITES = ["libero_10", "libero_spatial", "libero_object", "libero_goal"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_eval_output_dir(args):
    env_outdir = os.environ.get("LIBERO_PLUS_EVAL_OUTDIR", "").strip()
    if env_outdir != "":
        return ensure_dir(env_outdir)
    return ensure_dir(os.path.join(os.getcwd(), "eval_libero_plus", args.finetune_type))


def normalize_category(value):
    if value is None:
        return None
    return CATEGORY_ALIASES.get(str(value).strip().lower())


def parse_categories(category_value):
    categories = set()
    if category_value is None:
        return []

    if isinstance(category_value, str):
        normalized = normalize_category(category_value)
        if normalized is not None:
            categories.add(normalized)
    elif isinstance(category_value, list):
        for item in category_value:
            categories.update(parse_categories(item))
    elif isinstance(category_value, dict):
        for key, inner in category_value.items():
            normalized_key = normalize_category(key)
            if isinstance(inner, bool):
                if inner and normalized_key is not None:
                    categories.add(normalized_key)
            else:
                categories.update(parse_categories(inner))
                if normalized_key is not None and inner:
                    categories.add(normalized_key)
    return sorted(categories)


def load_task_classification(args):
    classification_path = os.path.join(
        args.libero_path,
        "libero",
        "libero",
        "benchmark",
        "task_classification.json",
    )
    if not os.path.exists(classification_path):
        raise FileNotFoundError(f"task_classification.json not found: {classification_path}")

    with open(classification_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    suite_to_name_to_categories = defaultdict(dict)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict in task_classification.json, got {type(raw)}")

    for suite_name, items in raw.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            task_name = item.get("name")
            if task_name is None:
                continue
            categories = parse_categories(item.get("category"))
            suite_to_name_to_categories[suite_name][task_name] = categories

    return suite_to_name_to_categories


def get_task_categories(classification_map, suite_name, task_name):
    return classification_map.get(suite_name, {}).get(task_name, [])


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def print_and_save_metadata_results(result_list, task_suite, classification_map, args):
    output_dir = get_eval_output_dir(args)
    ckpt_id = os.environ.get("LIBERO_PLUS_EVAL_CKPT_ID", "").strip()
    file_prefix = f"ckpt-{ckpt_id}_{args.finetune_type}" if ckpt_id != "" else args.finetune_type

    task_rows = []
    category_to_values = defaultdict(list)

    for task_id in range(task_num):
        this_result_list = result_list[task_id * num_eval_episodes: (task_id + 1) * num_eval_episodes]
        task = task_suite.get_task(task_id)
        task_name = task.name
        categories = get_task_categories(classification_map, args.finetune_type, task_name)

        if len(this_result_list) == 0:
            result_value = -1
        else:
            result_value = int(this_result_list[0][0])

        if result_value == 1:
            status = "success"
        elif result_value == 0:
            status = "fail"
        else:
            status = "skip"

        task_rows.append(
            {
                "suite": args.finetune_type,
                "task_id": task_id,
                "task_name": task_name,
                "result": result_value,
                "status": status,
                "categories": "|".join(categories),
            }
        )

        if result_value in [0, 1]:
            for category in categories:
                if category in CATEGORY_ORDER:
                    category_to_values[category].append(result_value)

    task_csv_path = os.path.join(output_dir, f"{file_prefix}_task_results.csv")
    write_csv(
        task_csv_path,
        ["suite", "task_id", "task_name", "result", "status", "categories"],
        task_rows,
    )

    category_rows = []
    for category in CATEGORY_ORDER:
        vals = category_to_values.get(category, [])
        category_rows.append(
            {
                "suite": args.finetune_type,
                "category": category,
                "num_tasks": len(vals),
                "avg_success": float(np.mean(vals)) if len(vals) > 0 else None,
            }
        )

    total_valid_results = [row["result"] for row in task_rows if row["result"] in [0, 1]]
    total_avg = float(np.mean(total_valid_results)) if len(total_valid_results) > 0 else None
    category_rows.append(
        {
            "suite": args.finetune_type,
            "category": "Total",
            "num_tasks": len(task_rows),
            "avg_success": total_avg,
        }
    )

    category_csv_path = os.path.join(output_dir, f"{file_prefix}_category_results.csv")
    write_csv(
        category_csv_path,
        ["suite", "category", "num_tasks", "avg_success"],
        category_rows,
    )

    print(f"[LIBERO-PLUS EVAL] saved task results: {task_csv_path}")
    print(f"[LIBERO-PLUS EVAL] saved category results: {category_csv_path}")

    print("\nBenchmark summary")
    header = ["Suite"] + CATEGORY_ORDER + ["Total"]
    row = [args.finetune_type]
    category_lookup = {item["category"]: item["avg_success"] for item in category_rows}
    for category in CATEGORY_ORDER:
        value = category_lookup.get(category)
        row.append("-" if value is None else f"{value * 100:.1f}")
    total_value = category_lookup.get("Total")
    row.append("-" if total_value is None else f"{total_value * 100:.1f}")
    print("\t".join(header))
    print("\t".join(row))


def evaluate_policy_ddp(args, model):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.finetune_type]()
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", device_id))

    visible_devices_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices_env.strip() != "":
        visible_device_ids = [int(x.strip()) for x in visible_devices_env.split(",") if x.strip() != ""]
        if local_rank >= len(visible_device_ids):
            raise ValueError(
                f"LOCAL_RANK={local_rank} is out of range for CUDA_VISIBLE_DEVICES={visible_devices_env}"
            )
        render_gpu_device_id = visible_device_ids[local_rank]
    else:
        render_gpu_device_id = local_rank

    if args.finetune_type not in SUPPORTED_SUITES:
        raise ValueError(f"Unsupported LIBERO-plus suite: {args.finetune_type}")

    classification_map = load_task_classification(args)

    global num_eval_episodes
    global task_num
    num_eval_episodes = 1
    task_num = task_suite.n_tasks

    num_sequences = num_eval_episodes * task_num
    all_eval_sequences = list(range(num_sequences))
    split_eval_sequences = np.array_split(all_eval_sequences, device_num)
    eval_sequences = split_eval_sequences[device_id].tolist()
    eval_sequences = list(eval_sequences)
    max_local_sequences = max(len(x) for x in split_eval_sequences)

    if device_id == 0:
        print(f"[LIBERO-PLUS EVAL] suite             : {args.finetune_type}")
        print(f"[LIBERO-PLUS EVAL] task_num          : {task_num}")
        print(f"[LIBERO-PLUS EVAL] num_eval_episodes : {num_eval_episodes}")
        print(f"[LIBERO-PLUS EVAL] local_rank        : {local_rank}")
        print(f"[LIBERO-PLUS EVAL] render_gpu_device : {render_gpu_device_id}")
        print(f"[LIBERO-PLUS EVAL] MUJOCO_GL         : {os.environ.get('MUJOCO_GL')}")
        print(f"[LIBERO-PLUS EVAL] PYOPENGL_PLATFORM : {os.environ.get('PYOPENGL_PLATFORM')}")

    results = []
    local_done = 0
    local_success = 0
    local_fail = 0
    local_skip = 0
    progress_print_every = 5

    for local_step in range(max_local_sequences):
        has_work = local_step < len(eval_sequences)

        if has_work:
            eval_id = eval_sequences[local_step]
            task_id = eval_id // num_eval_episodes
            exp_id = eval_id % num_eval_episodes
            task = task_suite.get_task(task_id)
            task_name = task.name
        else:
            eval_id = None
            task_id = None
            exp_id = None
            task = None
            task_name = None

        try:
            if has_work:
                task_bddl_file = os.path.join(
                    f"{args.libero_path}/libero/libero/bddl_files",
                    task.problem_folder,
                    task.bddl_file,
                )

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

                init_states = task_suite.get_task_init_states(task_id)
                init_state = init_states[exp_id]
                obs = env.set_init_state(init_state)

                for _ in range(5):
                    env.step(np.zeros(7))

                result = evaluate_libero_task(task, env, obs, args, model)
                results.append(result)

                local_done += 1
                if int(result) == 1:
                    local_success += 1
                else:
                    local_fail += 1

        except Exception as e:
            if has_work:
                local_done += 1
                local_skip += 1
                results.append(-1)
                print(
                    f"[LIBERO-PLUS EVAL][rank={device_id} gpu={render_gpu_device_id}] "
                    f"SKIP task_id={task_id} task_name={task_name} reason={type(e).__name__}: {e}",
                    flush=True,
                )

        sync_step = ((local_step + 1) % progress_print_every == 0) or (local_step == max_local_sequences - 1)

        if sync_step:
            should_print = has_work and (
                    (local_done % progress_print_every == 0) or (local_done == len(eval_sequences))
            )

            local_tensor = torch.tensor(
                [local_done, local_success, local_fail, local_skip, int(should_print)],
                device=f"cuda:{local_rank}",
                dtype=torch.long,
            )

            gathered = [torch.zeros_like(local_tensor) for _ in range(device_num)]
            torch.distributed.all_gather(gathered, local_tensor)

            if device_id == 0:
                any_rank_requested_print = any(int(x[4].item()) == 1 for x in gathered)
                if any_rank_requested_print:
                    total_done = sum(int(x[0].item()) for x in gathered)
                    total_success = sum(int(x[1].item()) for x in gathered)
                    total_fail = sum(int(x[2].item()) for x in gathered)
                    total_skip = sum(int(x[3].item()) for x in gathered)

                    per_rank = " | ".join(
                        [
                            f"rank{i}/gpu{visible_device_ids[i] if visible_devices_env.strip() != '' else i}: "
                            f"{int(g[0].item())}"
                            for i, g in enumerate(gathered)
                        ]
                    )

                    print(
                        f"[LIBERO-PLUS EVAL][GLOBAL] "
                        f"done={total_done}/{num_sequences} "
                        f"success={total_success} fail={total_fail} skip={total_skip} "
                        f"| {per_rank}",
                        flush=True,
                    )

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_tup_list.sort(key=lambda x: x[1])
        print_and_save_metadata_results(res_tup_list, task_suite, classification_map, args)


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
