"""Shared initialization, matched NFE3 training, and held-out interventions."""
from pathlib import Path
import random
import time

import numpy as np
import torch
from tqdm import trange

from methods.refresh_calibration.model import RefreshCalibratedCondition
from methods.latentloop.modules.action_aligned_joint import action_loss
from methods.latentloop.modules.trend_condition import scaled_mse
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save, load_native_v0_checkpoint
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    configure_strict_torch_determinism, load_frozen_simvla, freeze_module, move_batch, append_jsonl,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    _drop_unused_vlm, collate_exact_teacher_sequences,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import make_datasets, _balanced_indices
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import lr_factor, assert_frozen
from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
from tools.simvla.error_compensation_common import snapshots, identity, sha, write_json, read_json


def language_key(text):
    return " ".join(str(text).strip().replace("_", " ").split()).lower()


def path_for(c, variant, k, smoke=False):
    key = "bootstrap" if variant == "bootstrap" else f"{variant}_k{k}"
    return Path(c["output"]) / ("smoke_train" if smoke else "train") / key / "latest.pt"


def runtime(c):
    device = torch.device("cuda:0")
    configure_strict_torch_determinism(c["seed"])
    torch.set_num_threads(1)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total - 2 * 1024**3) / total)
    parent, payload = load_native_v0_checkpoint(c["condition_checkpoint"], device="cpu", require_final_150k=True)
    del parent
    data, heldout = make_datasets({**c, "training_k_c": 8}, payload)
    assets = snapshots(c)
    frozen, processor, action = load_frozen_simvla(checkpoint=assets["checkpoint_snapshot"],
        norm_stats=c["norm_stats"], smolvlm_model=assets["backbone_snapshot"], device=device)
    bank = {}
    # This lookup is instruction-only: it runs no vision encoder or language layers.
    texts = {data.store.query(w[0])["metadata"]["language_instruction"] for w in data.windows}
    texts |= {heldout.store.query(w[0])["metadata"]["language_instruction"] for w in heldout.windows}
    with torch.no_grad():
        embedding = frozen.vlm.model.text_model.get_input_embeddings()
        for text in sorted(texts):
            ids = processor.tokenizer(str(text), return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            bank[language_key(text)] = embedding(ids).float().mean(1).cpu()
    _drop_unused_vlm(frozen)
    del processor
    freeze_module(frozen)
    return device, frozen, action, data, heldout, bank


def sequence_at(data, index, device):
    batch = collate_exact_teacher_sequences([data[index]])
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import resize_with_pad_uint8
    images = batch["image_sequence"]
    if images.shape[-3:-1] != (224, 224):
        resized = np.stack([resize_with_pad_uint8(view.numpy(), 224)
                            for view in images.flatten(0, 2)])
        batch["image_sequence"] = torch.from_numpy(resized).reshape(*images.shape[:3], 224, 224, 3)
    # Match live preprocessing: CPU uint8 -> float32 /255 before CUDA transfer.
    batch["image_sequence"] = batch["image_sequence"].float() / 255.
    return move_batch(batch, device)


def language(batch, bank, device):
    return torch.cat([bank[language_key(t)] for t in batch["language_instruction"]]).to(device)


def select_queries(seed, step, length, k):
    rng = random.Random(seed * 10000000 + step)
    return [(rng.randrange(length), rng.randrange(1, k)) for _ in range(2)]


def prediction(model, batch, age, lang):
    state = model.prepare(batch["anchor_condition"].detach(), batch["image_sequence"][:, 0],
        batch["proprio_sequence"][:, 0], lang, batch["valid_mask"].bool(), batch["group_ids"])
    return model.predict(state, batch["image_sequence"][:, age], batch["proprio_sequence"][:, age]), state


def terms(c, model, action, batch, age, lang, bootstrap, grad):
    if bootstrap:
        predicted = model.direct(batch["image_sequence"][:, age], batch["proprio_sequence"][:, age],
                                 lang, batch["group_ids"])
    else:
        predicted, _ = prediction(model, batch, age, lang)
    target = batch["teacher_conditions"][:, age - 1].detach()
    cond = scaled_mse(predicted, target, batch["anchor_condition"], batch["valid_mask"].bool())
    act = predicted.new_zeros(())
    if not bootstrap:
        result = action.decode_action_from_condition(predicted, batch["proprio_sequence"][:, age],
            steps=3, initial_noise=batch["explicit_noises"][:, age - 1],
            requires_grad=grad, return_debug=True).final_action_latent
        teacher = action.action_space.normalize_action(batch["teacher_actions"][:, age - 1]).detach()
        act = action_loss(result, teacher)
    return act, cond, predicted


@torch.no_grad()
def validate(c, model, action, heldout, bank, directory, label, k, smoke, bootstrap):
    selected = _balanced_indices(heldout.identities, limit=2 if smoke else 30, seed=c["seed"])
    records = []
    for index in selected:
        batch = sequence_at(heldout, index, "cuda")
        lang = language(batch, bank, "cuda")
        # The wrong-scene intervention is held-out analysis only, never deployed.
        donor = next((i for i in selected if heldout.identities[i][0] != heldout.identities[index][0]), None)
        wrong = None
        if model.variant == "ridge" and donor is not None and not bootstrap:
            other = sequence_at(heldout, donor, "cuda")
            _, wrong = prediction(model, other, 1, language(other, bank, "cuda"))
        for age in (1, k - 1) if smoke else range(1, k):
            act, cond, predicted = terms(c, model, action, batch, age, lang, bootstrap, False)
            cosine = torch.nn.functional.cosine_similarity(predicted,
                batch["teacher_conditions"][:, age - 1].float(), dim=-1)
            item = dict(index=index, task_id=int(batch["task_id"][0]), age=age,
                condition_mse=float(cond),
                condition_cosine=float(cosine[batch["valid_mask"].bool()].mean()))
            if not bootstrap:
                item["action_l1"] = float(act)
            if wrong is not None:
                _, state = prediction(model, batch, age, lang)
                state.correction = wrong.correction
                model.update_reference(state)
                mismatch = model.predict(state, batch["image_sequence"][:, age], batch["proprio_sequence"][:, age])
                decoded = action.decode_action_from_condition(mismatch, batch["proprio_sequence"][:, age],
                    steps=3, initial_noise=batch["explicit_noises"][:, age - 1], return_debug=True).final_action_latent
                item["wrong_scene_action_l1"] = float(action_loss(decoded,
                    action.action_space.normalize_action(batch["teacher_actions"][:, age - 1])))
                item["wrong_scene_condition_mse"] = float(scaled_mse(mismatch,
                    batch["teacher_conditions"][:, age - 1], batch["anchor_condition"], batch["valid_mask"].bool()))
                item["donor_task_id"] = heldout.identities[donor][0]
            records.append(item)
    keys = ("action_l1", "condition_mse", "condition_cosine", "wrong_scene_action_l1", "wrong_scene_condition_mse")
    write_json(directory / f"heldout_{label}.json", dict(records=records,
        split=heldout.contract(), means={key: sum(r[key] for r in records if key in r) /
            sum(key in r for r in records) for key in keys if any(key in r for r in records)},
        scope="Held-out trajectory analysis; wrong-scene correction is an offline intervention, not online SR."))


def train(c, variant, k, smoke=False):
    wall_started = time.monotonic()
    bootstrap = variant == "bootstrap"
    run_id = identity(c)
    path = path_for(c, variant, k, smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = c["smoke_steps"] if smoke else c["bootstrap_steps"] if bootstrap else c["steps"]
    if (path.parent / "summary.json").exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        report = read_json(path.parent / "summary.json")
        if saved["identity"] != run_id or saved["step"] != total or sha(path) != report["checkpoint_sha256"]:
            raise RuntimeError("Completed checkpoint identity mismatch")
        return
    device, frozen, action, data, heldout, bank = runtime(c)
    torch.manual_seed(c["seed"])
    model = RefreshCalibratedCondition("fixed" if bootstrap else variant, **c["model"]).to(device)
    common_sha = None
    if not bootstrap:
        external = c.get("common_initialization")
        common = Path(external["path"]) if external else path_for(c, "bootstrap", 8, smoke)
        if external and sha(common) != external["sha256"]:
            raise RuntimeError("External common initialization checksum changed")
        saved = torch.load(common, map_location="cpu", weights_only=False)
        expected = external["step"] if external else c["smoke_steps"] if smoke else c["bootstrap_steps"]
        expected_identity = external["identity"] if external else run_id
        if saved["identity"] != expected_identity or saved["step"] != expected or saved["variant"] != "bootstrap":
            raise RuntimeError("Shared initialization is not complete")
        if set(saved["language_bank"]) != set(bank) or any(not torch.equal(saved["language_bank"][t], bank[t]) for t in bank):
            raise RuntimeError("Frozen instruction embeddings changed")
        model.initialize_common(saved["model"])
        common_sha = sha(common)
    initial = state_hash(model)
    frozen_initial = state_hash(frozen)
    contract = dict(identity=run_id, variant=variant, training_k_c=k, nfe=3, generation_loop=False,
        steps=total, common_checkpoint_sha256=common_sha, initial_model_sha256=initial,
        train=data.contract(), heldout=heldout.contract(), model=c["model"],
        trainable_parameters=sum(p.numel() for p in model.parameters()), batch_size=2,
        optimizer="AdamW lr1e-4 wd0; warmup150 then cosine to0.1x; clip1",
        target="Frozen original Condition and original NFE10 continuous actions with identical explicit noise",
        objective="normalized Condition MSE" if bootstrap else "first5 continuous-action L1 + 0.05 normalized Condition MSE",
        language="Frozen original token embeddings, instruction-only mean; no language transformer execution",
        padding="All arms copy invalid anchor positions; only valid positions participate in loss and ridge fit",
        feature_initialization="Same shared bootstrap, independently optimized in each arm; identical architecture, not tied weights",
        image_preprocessing="Native cached RGB -> official-client 224px PIL bilinear pad -> float32/255 -> 64px model encoder",
        adaptive_state="Ridge fit uses refresh observation and original Condition only; no future teacher in predict")
    write_json(path.parent / "training_contract.json", contract)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda n: lr_factor(n, total, min(c["warmup_steps"], max(1, total // 10))))
    start, elapsed = 0, 0.
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved["contract"] != contract:
            raise RuntimeError("Resume contract changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, elapsed = saved["step"], saved["training_seconds"]
    if not start:
        with torch.no_grad():
            sample = sequence_at(data, 0, device)
            teacher = action.decode_action_from_condition(sample["teacher_conditions"][:, 0],
                sample["proprio_sequence"][:, 1], steps=10, initial_noise=sample["explicit_noises"][:, 0])
            diff = float((teacher - sample["teacher_actions"][:, 0]).abs().max())
            if diff > 2e-4:
                raise RuntimeError(f"Original teacher/cache mismatch: {diff}")
            write_json(path.parent / "teacher_parity.json", dict(verdict="PASS", max_action_difference=diff))
        validate(c, model, action, heldout, bank, path.parent, "before", k, smoke, bootstrap)
    tracker = None
    if not smoke and start < total and c.get("wandb_project"):
        try:
            import wandb
            tracker = wandb.init(project=c["wandb_project"], name=f"refresh_calibration_{variant}_k{k}",
                id=run_id[:12] + f"_{variant}_k{k}", resume="allow", config=contract,
                dir=str(path.parent), settings=wandb.Settings(init_timeout=20))
        except Exception as exc:
            print(f"WANDB_WARNING {exc}", flush=True)
    begun = time.monotonic()
    try:
        progress = trange(start + 1, total + 1, desc=f"{variant} K{k} NFE3", mininterval=2)
        for step in progress:
            optimizer.zero_grad(set_to_none=True)
            values = dict(action_l1=0., condition_mse=0.)
            for index, age in select_queries(c["seed"], step, len(data), k):
                batch = sequence_at(data, index, device)
                act, cond, _ = terms(c, model, action, batch, age, language(batch, bank, device), bootstrap, True)
                loss = cond if bootstrap else act + c["condition_weight"] * cond
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss; checkpoint preserved")
                (loss / 2).backward()
                values["action_l1"] += float(act.detach()) / 2
                values["condition_mse"] += float(cond.detach()) / 2
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if not norm > 0:
                raise RuntimeError("No trainable gradients")
            assert_frozen(frozen)
            optimizer.step()
            scheduler.step()
            seconds = elapsed + time.monotonic() - begun
            progress.set_postfix(act=f"{values['action_l1']:.4f}", cond=f"{values['condition_mse']:.4f}")
            if step in (1, total) or step % c["log_interval"] == 0:
                record = dict(step=step, **values, lr=optimizer.param_groups[0]["lr"],
                    training_seconds=seconds, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    estimated_remaining_seconds=(total-step)*(time.monotonic()-begun)/(step-start))
                append_jsonl(path.parent / "metrics.jsonl", record)
                if tracker:
                    try: tracker.log(record, step=step)
                    except Exception as exc: print(f"WANDB_WARNING {exc}", flush=True)
            if step == total or step % c["save_interval"] == 0:
                atomic_torch_save(dict(format="simvla_refresh_calibration_v1", identity=run_id,
                    variant=variant, k_c=k, step=step, contract=contract, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    training_seconds=seconds, language_bank=bank), path)
                print(f"CHECKPOINT step={step} path={path}", flush=True)
    finally:
        if tracker:
            try: tracker.finish()
            except Exception: pass
    if state_hash(frozen) != frozen_initial or state_hash(model) == initial:
        raise RuntimeError("Frozen teacher changed or trainable model unchanged")
    validate(c, model, action, heldout, bank, path.parent, "after", k, smoke, bootstrap)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    write_json(path.parent / "summary.json", dict(identity=run_id, verdict="TRAIN_COMPLETE", steps=total,
        variant=variant, k_c=k, nfe=3, checkpoint=str(path), checkpoint_sha256=sha(path),
        training_seconds=saved["training_seconds"], parameters=contract["trainable_parameters"],
        invocation_wall_seconds=time.monotonic()-wall_started,
        frozen_teacher_unchanged=True, common_checkpoint_sha256=common_sha))
