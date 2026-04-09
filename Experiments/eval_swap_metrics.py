# Experiments/eval_swap_metrics.py

import os
import sys
import json
import yaml
import argparse
from typing import Dict, Any, Optional, List

import numpy as np
import torch

# 项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AuxClassifier.sequence_classifier import SequenceClassifier


def load_yaml(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_config = os.path.join(project_root, "config.yaml")
    default_stage1_root = os.path.join(os.path.dirname(__file__), "stage1_swap_results")
    default_stage2_root = os.path.join(os.path.dirname(__file__), "stage2_swap_results")
    default_output_root = os.path.join(os.path.dirname(__file__), "swap_metrics")

    parser = argparse.ArgumentParser(description="Evaluate swap metrics for stage1/stage2 results.")
    parser.add_argument("--config", type=str, default=default_config, help="Path to config.yaml")

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "stage1", "stage2"],
        help="Which stage to evaluate"
    )
    parser.add_argument(
        "--pair_type",
        type=str,
        default="all",
        choices=["all", "text_emotion", "text_intensity", "identity"],
        help="Which pair type to evaluate"
    )

    parser.add_argument("--stage1_root", type=str, default=default_stage1_root, help="Root dir of stage1 swap results")
    parser.add_argument("--stage2_root", type=str, default=default_stage2_root, help="Root dir of stage2 swap results")
    parser.add_argument("--output_root", type=str, default=default_output_root, help="Where to save metrics json")

    parser.add_argument(
        "--emotion_ckpt",
        type=str,
        default=os.path.join(project_root, "AuxClassifier", "checkpoints_emotion", "emotion", "model_best.pth"),
        help="Path to emotion classifier checkpoint"
    )
    parser.add_argument(
        "--intensity_ckpt",
        type=str,
        default=os.path.join(project_root, "AuxClassifier", "checkpoints_emotion", "intensity", "model_best.pth"),
        help="Path to intensity classifier checkpoint"
    )
    parser.add_argument(
        "--identity_ckpt",
        type=str,
        default=os.path.join(project_root, "AuxClassifier", "checkpoints_identity", "model_best.pth"),
        help="Path to identity classifier checkpoint"
    )

    parser.add_argument("--gpu", type=int, default=None, help="Override gpu id")
    parser.add_argument("--save_per_group", action="store_true", help="Whether to save detailed per-group metrics")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_npy(path: str) -> np.ndarray:
    return np.load(path)


def tensorize(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def build_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    num_classes = ckpt["num_classes"]

    model = SequenceClassifier(
        input_dim=103,
        hidden_dim=args["hidden_dim"],
        num_heads=args["num_heads"],
        num_layers=args["num_layers"],
        ff_dim=args["ff_dim"],
        num_classes=num_classes,
        dropout=args["dropout"],
        max_len=args["max_len"],
        use_cls_token=args.get("use_cls_token", False),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    info = {
        "model": model,
        "num_classes": num_classes,
        "label_type": ckpt.get("label_type", None),
        "label_names": ckpt.get("label_names", None),
        "checkpoint_path": ckpt_path,
    }
    return info


def infer_mask_from_gt(exp_gt: np.ndarray, jaw_gt: np.ndarray) -> np.ndarray:
    """
    用 GT 的非零帧推断有效 mask。
    输入通常形状为 [1, T, D]。
    输出形状为 [1, T]，float32，取值 0/1。
    """
    if exp_gt.ndim == 3:
        exp_abs = np.abs(exp_gt).sum(axis=-1)   # [1, T]
    else:
        raise ValueError(f"exp_gt should be [1,T,D], got {exp_gt.shape}")

    if jaw_gt.ndim == 3:
        jaw_abs = np.abs(jaw_gt).sum(axis=-1)   # [1, T]
    else:
        raise ValueError(f"jaw_gt should be [1,T,D], got {jaw_gt.shape}")

    valid = (exp_abs + jaw_abs) > 0
    return valid.astype(np.float32)


@torch.no_grad()
def predict_label(
    model: SequenceClassifier,
    exp_arr: np.ndarray,
    jaw_arr: np.ndarray,
    mask_arr: np.ndarray,
    device: torch.device,
) -> int:
    exp = tensorize(exp_arr, device)
    jaw = tensorize(jaw_arr, device)
    mask = tensorize(mask_arr, device)

    logits = model.forward_from_exp_jaw(
        exp=exp,
        jaw=jaw,
        mask=mask,
        return_features=False,
        return_attn=False
    )
    pred = torch.argmax(logits, dim=1).item()
    return int(pred)


def argmax_one_hot(arr: np.ndarray) -> int:
    return int(np.argmax(arr))


def to_bool_int(x: bool) -> int:
    return 1 if bool(x) else 0


def compute_jpe(pred_jaw: np.ndarray, gt_jaw: np.ndarray, mask: np.ndarray) -> float:
    """
    stage2 专用：Jaw Preservation Error
    比较交换后结果与源音频对应 GT 的 jaw 参数差异。
    只在有效帧上计算。
    """
    # [1,T,3]
    diff = pred_jaw - gt_jaw
    dist = np.linalg.norm(diff, axis=-1)  # [1,T]
    valid = mask > 0

    if valid.sum() == 0:
        return 0.0
    return float(dist[valid].mean())


def compute_group_metrics(
    stage: str,
    pair_type: str,
    group_dir: str,
    models: Dict[str, Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    meta_path = os.path.join(group_dir, "meta.json")
    meta = load_json(meta_path)

    # 读取 GT
    source_gt_exp = load_npy(os.path.join(group_dir, "source_gt_exp.npy"))
    source_gt_jaw = load_npy(os.path.join(group_dir, "source_gt_jaw.npy"))
    target_gt_exp = load_npy(os.path.join(group_dir, "target_gt_exp.npy"))
    target_gt_jaw = load_npy(os.path.join(group_dir, "target_gt_jaw.npy"))

    source_mask = infer_mask_from_gt(source_gt_exp, source_gt_jaw)
    target_mask = infer_mask_from_gt(target_gt_exp, target_gt_jaw)

    # 读取交换输出
    source_swap_exp = load_npy(os.path.join(group_dir, "source_swap_exp.npy"))
    source_swap_jaw = load_npy(os.path.join(group_dir, "source_swap_jaw.npy"))
    target_swap_exp = load_npy(os.path.join(group_dir, "target_swap_exp.npy"))
    target_swap_jaw = load_npy(os.path.join(group_dir, "target_swap_jaw.npy"))

    emo_model = models["emotion"]["model"]
    int_model = models["intensity"]["model"]
    id_model = models["identity"]["model"]

    # 预测：A <- B
    pred_source_swap_emotion = predict_label(emo_model, source_swap_exp, source_swap_jaw, source_mask, device)
    pred_source_swap_intensity = predict_label(int_model, source_swap_exp, source_swap_jaw, source_mask, device)
    pred_source_swap_identity = predict_label(id_model, source_swap_exp, source_swap_jaw, source_mask, device)

    # 预测：B <- A
    pred_target_swap_emotion = predict_label(emo_model, target_swap_exp, target_swap_jaw, target_mask, device)
    pred_target_swap_intensity = predict_label(int_model, target_swap_exp, target_swap_jaw, target_mask, device)
    pred_target_swap_identity = predict_label(id_model, target_swap_exp, target_swap_jaw, target_mask, device)

    if pair_type == "text_emotion":
        # 元信息：source_emotion/target_emotion/source_intensity/target_intensity/person_id
        source_e = meta["source_emotion"]
        target_e = meta["target_emotion"]
        source_r = meta["source_intensity"]
        target_r = meta["target_intensity"]
        pid = meta["person_id"]

        emotion_names = models["emotion"]["label_names"]
        intensity_names = models["intensity"]["label_names"]
        identity_names = models["identity"]["label_names"]

        source_e_idx = emotion_names.index(source_e)
        target_e_idx = emotion_names.index(target_e)
        source_r_idx = intensity_names.index(source_r)
        target_r_idx = intensity_names.index(target_r)
        pid_idx = identity_names.index(pid)

        # A <- B
        dir1 = {
            "EmTA": to_bool_int(pred_source_swap_emotion == target_e_idx),
            "IdPA": to_bool_int(pred_source_swap_identity == pid_idx),
            "InPA": to_bool_int(pred_source_swap_intensity == source_r_idx),
        }
        dir1["SSR_emo"] = to_bool_int(dir1["EmTA"] and dir1["IdPA"] and dir1["InPA"])

        # B <- A
        dir2 = {
            "EmTA": to_bool_int(pred_target_swap_emotion == source_e_idx),
            "IdPA": to_bool_int(pred_target_swap_identity == pid_idx),
            "InPA": to_bool_int(pred_target_swap_intensity == target_r_idx),
        }
        dir2["SSR_emo"] = to_bool_int(dir2["EmTA"] and dir2["IdPA"] and dir2["InPA"])

        metrics = average_two_dirs(dir1, dir2)

    elif pair_type == "text_intensity":
        # 元信息：source_emotion/target_emotion(应相同)/source_intensity/target_intensity/person_id
        source_e = meta["source_emotion"]
        target_e = meta["target_emotion"]
        source_r = meta["source_intensity"]
        target_r = meta["target_intensity"]
        pid = meta["person_id"]

        emotion_names = models["emotion"]["label_names"]
        intensity_names = models["intensity"]["label_names"]
        identity_names = models["identity"]["label_names"]

        source_e_idx = emotion_names.index(source_e)
        target_e_idx = emotion_names.index(target_e)
        source_r_idx = intensity_names.index(source_r)
        target_r_idx = intensity_names.index(target_r)
        pid_idx = identity_names.index(pid)

        # A <- B
        dir1 = {
            "InTA": to_bool_int(pred_source_swap_intensity == target_r_idx),
            "EmPA": to_bool_int(pred_source_swap_emotion == source_e_idx),
            "IdPA": to_bool_int(pred_source_swap_identity == pid_idx),
        }
        dir1["SSR_int"] = to_bool_int(dir1["InTA"] and dir1["EmPA"] and dir1["IdPA"])

        # B <- A
        dir2 = {
            "InTA": to_bool_int(pred_target_swap_intensity == source_r_idx),
            "EmPA": to_bool_int(pred_target_swap_emotion == target_e_idx),
            "IdPA": to_bool_int(pred_target_swap_identity == pid_idx),
        }
        dir2["SSR_int"] = to_bool_int(dir2["InTA"] and dir2["EmPA"] and dir2["IdPA"])

        metrics = average_two_dirs(dir1, dir2)

    elif pair_type == "identity":
        # 元信息：source_person_id/target_person_id/emotion/intensity
        source_pid = meta["source_person_id"]
        target_pid = meta["target_person_id"]
        emotion = meta["emotion"]
        intensity = meta["intensity"]

        emotion_names = models["emotion"]["label_names"]
        intensity_names = models["intensity"]["label_names"]
        identity_names = models["identity"]["label_names"]

        source_pid_idx = identity_names.index(source_pid)
        target_pid_idx = identity_names.index(target_pid)
        e_idx = emotion_names.index(emotion)
        r_idx = intensity_names.index(intensity)

        # A <- B
        dir1 = {
            "IdTA": to_bool_int(pred_source_swap_identity == target_pid_idx),
            "EmPA": to_bool_int(pred_source_swap_emotion == e_idx),
            "InPA": to_bool_int(pred_source_swap_intensity == r_idx),
        }
        dir1["SSR_id"] = to_bool_int(dir1["IdTA"] and dir1["EmPA"] and dir1["InPA"])

        # B <- A
        dir2 = {
            "IdTA": to_bool_int(pred_target_swap_identity == source_pid_idx),
            "EmPA": to_bool_int(pred_target_swap_emotion == e_idx),
            "InPA": to_bool_int(pred_target_swap_intensity == r_idx),
        }
        dir2["SSR_id"] = to_bool_int(dir2["IdTA"] and dir2["EmPA"] and dir2["InPA"])

        metrics = average_two_dirs(dir1, dir2)

    else:
        raise ValueError(f"Unsupported pair_type: {pair_type}")

    # stage2 额外加语音保持指标：JPE
    if stage == "stage2":
        source_jpe = compute_jpe(source_swap_jaw, source_gt_jaw, source_mask)
        target_jpe = compute_jpe(target_swap_jaw, target_gt_jaw, target_mask)
        metrics["JPE"] = float((source_jpe + target_jpe) / 2.0)

    result = {
        "group_dir": group_dir,
        "pair_type": pair_type,
        "stage": stage,
        "metrics": metrics,
        "meta": meta,
    }
    return result


def average_two_dirs(dir1: Dict[str, float], dir2: Dict[str, float]) -> Dict[str, float]:
    out = {}
    keys = sorted(set(dir1.keys()) | set(dir2.keys()))
    for k in keys:
        v1 = float(dir1.get(k, 0.0))
        v2 = float(dir2.get(k, 0.0))
        out[k] = (v1 + v2) / 2.0
    return out


def list_group_dirs(root_dir: str) -> List[str]:
    if not os.path.exists(root_dir):
        return []
    group_dirs = []
    for name in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and name.startswith("group_"):
            group_dirs.append(full)
    return group_dirs


def aggregate_group_metrics(group_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(group_results) == 0:
        return {"num_groups": 0, "mean_metrics": {}}

    keys = set()
    for g in group_results:
        keys.update(g["metrics"].keys())

    mean_metrics = {}
    for k in sorted(keys):
        vals = [float(g["metrics"][k]) for g in group_results if k in g["metrics"]]
        mean_metrics[k] = float(np.mean(vals)) if len(vals) > 0 else 0.0

    return {
        "num_groups": len(group_results),
        "mean_metrics": mean_metrics
    }


def evaluate_one_setting(
    stage: str,
    pair_type: str,
    result_root: str,
    models: Dict[str, Dict[str, Any]],
    device: torch.device,
    save_root: str,
    save_per_group: bool = False,
):
    group_dirs = list_group_dirs(result_root)
    print("=" * 80)
    print(f"Evaluating {stage} / {pair_type}")
    print(f"result_root: {result_root}")
    print(f"num_groups : {len(group_dirs)}")
    print("=" * 80)

    group_results = []
    for group_dir in group_dirs:
        try:
            group_result = compute_group_metrics(
                stage=stage,
                pair_type=pair_type,
                group_dir=group_dir,
                models=models,
                device=device,
            )
            group_results.append(group_result)
        except Exception as e:
            print(f"[Skip] {group_dir} because of error: {e}")

    summary = aggregate_group_metrics(group_results)
    summary.update({
        "stage": stage,
        "pair_type": pair_type,
        "result_root": result_root,
        "classifier_checkpoints": {
            "emotion": models["emotion"]["checkpoint_path"],
            "intensity": models["intensity"]["checkpoint_path"],
            "identity": models["identity"]["checkpoint_path"],
        }
    })

    out_dir = os.path.join(save_root, stage)
    ensure_dir(out_dir)
    save_json(os.path.join(out_dir, f"{pair_type}_summary.json"), summary)

    if save_per_group:
        save_json(os.path.join(out_dir, f"{pair_type}_per_group.json"), group_results)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    args = parse_args()
    config = load_yaml(args.config)

    device = torch.device(
        f"cuda:{args.gpu if args.gpu is not None else config['predict']['gpu']}"
        if torch.cuda.is_available() else "cpu"
    )

    models = {
        "emotion": build_model_from_checkpoint(args.emotion_ckpt, device),
        "intensity": build_model_from_checkpoint(args.intensity_ckpt, device),
        "identity": build_model_from_checkpoint(args.identity_ckpt, device),
    }

    # identity checkpoint 没有显式 label_names 时，从 args 里的数据集类推不太稳，
    # 因此前面 train_identity.py 需要保存 label_names。
    # 若旧 checkpoint 没有，则这里给出更明确的报错。
    if models["identity"]["label_names"] is None:
        raise ValueError(
            "Identity checkpoint does not contain label_names. "
            "Please retrain train_identity.py with the updated version that saves label_names."
        )

    if args.stage == "all":
        stages = ["stage1", "stage2"]
    else:
        stages = [args.stage]

    if args.pair_type == "all":
        pair_types = ["text_emotion", "text_intensity", "identity"]
    else:
        pair_types = [args.pair_type]

    all_summaries = []

    for stage in stages:
        stage_root = args.stage1_root if stage == "stage1" else args.stage2_root

        for pair_type in pair_types:
            result_root = os.path.join(stage_root, pair_type)
            if not os.path.exists(result_root):
                print(f"[Skip] result_root does not exist: {result_root}")
                continue

            summary = evaluate_one_setting(
                stage=stage,
                pair_type=pair_type,
                result_root=result_root,
                models=models,
                device=device,
                save_root=args.output_root,
                save_per_group=args.save_per_group,
            )
            all_summaries.append(summary)

    save_json(os.path.join(args.output_root, "all_summaries.json"), all_summaries)

    print("=" * 80)
    print("Evaluation finished.")
    print(f"Saved to: {args.output_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()