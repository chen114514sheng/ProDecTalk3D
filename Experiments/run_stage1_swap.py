# Experiments/run_stage1_swap.py

import os
import sys
import json
import yaml
import argparse
from typing import Dict, Any, Tuple

import numpy as np
import torch

# 项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DataProcess.Dataload import CustomDataset
from VQVAE2.VQVAE import VQVAE


def load_yaml(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_config = os.path.join(project_root, "config.yaml")
    default_pair_dir = os.path.join(os.path.dirname(__file__), "swap_pairs")
    default_save_root = os.path.join(os.path.dirname(__file__), "stage1_swap_results")

    parser = argparse.ArgumentParser(description="Run stage1 swap experiments for VQ-VAE reconstruction stage.")
    parser.add_argument("--config", type=str, default=default_config, help="Path to config.yaml")
    parser.add_argument(
        "--pair_type",
        type=str,
        default="text_emotion",
        choices=["text_emotion", "text_intensity", "identity"],
        help="Which pair set to run"
    )
    parser.add_argument(
        "--pair_json",
        type=str,
        default=None,
        help="Path to pair json; if None, auto use Experiments/swap_pairs/<pair_type>_pairs.json"
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=default_save_root,
        help="Root directory to save stage1 swap outputs"
    )
    parser.add_argument("--gpu", type=int, default=None, help="Override gpu id")
    parser.add_argument("--max_groups", type=int, default=-1, help="Maximum number of pair groups to run, -1 means all")
    parser.add_argument(
        "--deduplicate_reverse",
        action="store_true",
        help="Skip reverse-duplicate pairs, only keep one group for {A,B}"
    )
    return parser.parse_args()


def get_stage1_ckpt_path(config: Dict) -> str:
    """
    优先使用 predict.vqvae2_dir；
    若不存在，再回退到 stage1.checkpoint_dir/model_val.pth
    """
    ckpt_path = config.get("predict", {}).get("vqvae2_dir", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        return ckpt_path

    stage1_ckpt_dir = config["stage1"]["checkpoint_dir"]
    fallback = os.path.join(stage1_ckpt_dir, "model_val.pth")
    if os.path.exists(fallback):
        return fallback

    raise FileNotFoundError(
        f"Cannot find stage1 checkpoint. "
        f"Tried predict.vqvae2_dir={ckpt_path} and fallback={fallback}"
    )


def build_model(config: Dict, device: torch.device) -> VQVAE:
    model = VQVAE(
        config["stage1"]["embed_dim"],
        config["stage1"]["num_heads"],
        config["stage1"]["num_layers_style"],
        config["stage1"]["num_layers_top"],
        config["stage1"]["num_layers_bottom"],
        config["stage1"]["num_embeddings"],
    ).to(device)

    ckpt_path = get_stage1_ckpt_path(config)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    print("=" * 80)
    print("Loaded stage1 model checkpoint:")
    print(ckpt_path)
    print("=" * 80)

    return model


def build_dataset_and_index(data_dir: str):
    dataset = CustomDataset(data_dir)

    token_to_index = {}
    for idx, file_path in enumerate(dataset.files):
        video_token = dataset.extract_video_token(file_path)
        if video_token not in token_to_index:
            token_to_index[video_token] = idx

    return dataset, token_to_index


def ensure_tensor(x, device: torch.device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)


def prepare_text(text):
    """
    与训练时 DataLoader 的 batch 输出保持一致：
    batch_size=1 时，训练里 text 通常是长度为1的 list[str]
    """
    if isinstance(text, str):
        return [text]
    if isinstance(text, (list, tuple)):
        return list(text)
    return [str(text)]


def load_single_sample(dataset: CustomDataset, index: int, device: torch.device) -> Dict[str, Any]:
    """
    dataset.__getitem__ 返回：
    video_token, person_one_hot, emotion_one_hot, text, audio, shape_data, exp_data, jaw_data, mask
    """
    video_token, person_one_hot, emotion_one_hot, text, audio, shape_data, exp_data, jaw_data, mask = dataset[index]

    sample = {
        "video_token": video_token,
        "person_one_hot": ensure_tensor(person_one_hot, device).unsqueeze(0),
        "emotion_one_hot": ensure_tensor(emotion_one_hot, device).unsqueeze(0),
        "text": prepare_text(text),
        "audio": audio,  # stage1 不用
        "shape_data": ensure_tensor(shape_data, device).unsqueeze(0),
        "exp_data": ensure_tensor(exp_data, device).unsqueeze(0),
        "jaw_data": ensure_tensor(jaw_data, device).unsqueeze(0),
        "mask": ensure_tensor(mask, device).unsqueeze(0),
    }
    return sample


@torch.no_grad()
def reconstruct_original(model: VQVAE, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    重建原输出：源条件不变
    """
    if hasattr(model, "reconstruct_with_swapped_condition"):
        exp_out, jaw_out = model.reconstruct_with_swapped_condition(
            source_person_one_hot=sample["person_one_hot"],
            source_text=sample["text"],
            source_exp=sample["exp_data"],
            source_jaw=sample["jaw_data"],
        )
        return exp_out, jaw_out

    if hasattr(model, "encode_codes") and hasattr(model, "decode_codes"):
        q_top, q_bottom = model.encode_codes(
            sample["person_one_hot"],
            sample["text"],
            sample["exp_data"],
            sample["jaw_data"],
        )
        exp_out, jaw_out = model.decode_codes(
            sample["person_one_hot"],
            sample["text"],
            q_top,
            q_bottom,
        )
        return exp_out, jaw_out

    raise AttributeError(
        "VQVAE model must provide reconstruct_with_swapped_condition "
        "or encode_codes/decode_codes for stage1 swap experiment."
    )


@torch.no_grad()
def reconstruct_swapped(
    model: VQVAE,
    source_sample: Dict[str, Any],
    target_sample: Dict[str, Any],
    pair_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    交换原则：
    1) text_emotion / text_intensity: 固定源编码特征和源身份，只换目标文本
    2) identity: 固定源编码特征和源文本，只换目标身份
    """
    target_person = None
    target_text = None

    if pair_type in ["text_emotion", "text_intensity"]:
        target_text = target_sample["text"]
    elif pair_type == "identity":
        target_person = target_sample["person_one_hot"]
    else:
        raise ValueError(f"Unsupported pair_type: {pair_type}")

    if hasattr(model, "reconstruct_with_swapped_condition"):
        exp_out, jaw_out = model.reconstruct_with_swapped_condition(
            source_person_one_hot=source_sample["person_one_hot"],
            source_text=source_sample["text"],
            source_exp=source_sample["exp_data"],
            source_jaw=source_sample["jaw_data"],
            target_person_one_hot=target_person,
            target_text=target_text,
        )
        return exp_out, jaw_out

    if hasattr(model, "encode_codes") and hasattr(model, "decode_codes"):
        q_top, q_bottom = model.encode_codes(
            source_sample["person_one_hot"],
            source_sample["text"],
            source_sample["exp_data"],
            source_sample["jaw_data"],
        )

        decode_person = source_sample["person_one_hot"] if target_person is None else target_person
        decode_text = source_sample["text"] if target_text is None else target_text

        exp_out, jaw_out = model.decode_codes(
            decode_person,
            decode_text,
            q_top,
            q_bottom,
        )
        return exp_out, jaw_out

    raise AttributeError(
        "VQVAE model must provide reconstruct_with_swapped_condition "
        "or encode_codes/decode_codes for stage1 swap experiment."
    )


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def save_array(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def save_group_results(
    save_dir: str,
    pair_meta: Dict[str, Any],
    source_sample: Dict[str, Any],
    target_sample: Dict[str, Any],
    source_rec: Tuple[torch.Tensor, torch.Tensor],
    target_rec: Tuple[torch.Tensor, torch.Tensor],
    source_swap: Tuple[torch.Tensor, torch.Tensor],
    target_swap: Tuple[torch.Tensor, torch.Tensor],
):
    os.makedirs(save_dir, exist_ok=True)

    # 保存 GT
    save_array(os.path.join(save_dir, "source_gt_exp.npy"), tensor_to_numpy(source_sample["exp_data"]))
    save_array(os.path.join(save_dir, "source_gt_jaw.npy"), tensor_to_numpy(source_sample["jaw_data"]))
    save_array(os.path.join(save_dir, "target_gt_exp.npy"), tensor_to_numpy(target_sample["exp_data"]))
    save_array(os.path.join(save_dir, "target_gt_jaw.npy"), tensor_to_numpy(target_sample["jaw_data"]))

    # 保存原重建
    save_array(os.path.join(save_dir, "source_rec_exp.npy"), tensor_to_numpy(source_rec[0]))
    save_array(os.path.join(save_dir, "source_rec_jaw.npy"), tensor_to_numpy(source_rec[1]))
    save_array(os.path.join(save_dir, "target_rec_exp.npy"), tensor_to_numpy(target_rec[0]))
    save_array(os.path.join(save_dir, "target_rec_jaw.npy"), tensor_to_numpy(target_rec[1]))

    # 保存交换重建
    save_array(os.path.join(save_dir, "source_swap_exp.npy"), tensor_to_numpy(source_swap[0]))
    save_array(os.path.join(save_dir, "source_swap_jaw.npy"), tensor_to_numpy(source_swap[1]))
    save_array(os.path.join(save_dir, "target_swap_exp.npy"), tensor_to_numpy(target_swap[0]))
    save_array(os.path.join(save_dir, "target_swap_jaw.npy"), tensor_to_numpy(target_swap[1]))

    meta = dict(pair_meta)
    meta.update({
        "source_text": source_sample["text"][0] if len(source_sample["text"]) > 0 else "",
        "target_text": target_sample["text"][0] if len(target_sample["text"]) > 0 else "",
        "saved_items": [
            "source_gt_exp.npy", "source_gt_jaw.npy",
            "target_gt_exp.npy", "target_gt_jaw.npy",
            "source_rec_exp.npy", "source_rec_jaw.npy",
            "target_rec_exp.npy", "target_rec_jaw.npy",
            "source_swap_exp.npy", "source_swap_jaw.npy",
            "target_swap_exp.npy", "target_swap_jaw.npy",
        ]
    })

    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def infer_pair_json(pair_type: str, pair_json: str = None) -> str:
    if pair_json is not None:
        return pair_json
    return os.path.join(os.path.dirname(__file__), "swap_pairs", f"{pair_type}_pairs.json")


def canonical_pair_key(pair: Dict[str, Any]):
    """
    用于去掉 A->B 与 B->A 的反向重复。
    """
    a = pair["source_video_token"]
    b = pair["target_video_token"]
    return tuple(sorted([a, b]))


def main():
    args = parse_args()
    config = load_yaml(args.config)

    device = torch.device(
        f"cuda:{args.gpu if args.gpu is not None else config['predict']['gpu']}"
        if torch.cuda.is_available() else "cpu"
    )

    pair_json_path = infer_pair_json(args.pair_type, args.pair_json)
    with open(pair_json_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    test_dir = config["test_file_path"]
    dataset, token_to_index = build_dataset_and_index(test_dir)
    model = build_model(config, device)

    save_root = os.path.join(args.save_root, args.pair_type)
    os.makedirs(save_root, exist_ok=True)

    seen = set()
    run_count = 0

    for pair_idx, pair in enumerate(pairs):
        if args.deduplicate_reverse:
            key = canonical_pair_key(pair)
            if key in seen:
                continue
            seen.add(key)

        source_token = pair["source_video_token"]
        target_token = pair["target_video_token"]

        if source_token not in token_to_index or target_token not in token_to_index:
            print(f"[Skip] token not found in dataset: {source_token} or {target_token}")
            continue

        source_index = token_to_index[source_token]
        target_index = token_to_index[target_token]

        source_sample = load_single_sample(dataset, source_index, device)
        target_sample = load_single_sample(dataset, target_index, device)

        # 原重建
        source_rec = reconstruct_original(model, source_sample)
        target_rec = reconstruct_original(model, target_sample)

        # 双向交换重建
        # source_swap: A <- B
        source_swap = reconstruct_swapped(model, source_sample, target_sample, args.pair_type)
        # target_swap: B <- A
        target_swap = reconstruct_swapped(model, target_sample, source_sample, args.pair_type)

        group_name = f"group_{run_count:05d}"
        group_save_dir = os.path.join(save_root, group_name)

        save_group_results(
            save_dir=group_save_dir,
            pair_meta=pair,
            source_sample=source_sample,
            target_sample=target_sample,
            source_rec=source_rec,
            target_rec=target_rec,
            source_swap=source_swap,
            target_swap=target_swap,
        )

        print(f"[Saved] {group_save_dir}")
        run_count += 1

        if args.max_groups > 0 and run_count >= args.max_groups:
            break

    summary = {
        "pair_type": args.pair_type,
        "pair_json": pair_json_path,
        "test_dir": test_dir,
        "save_root": save_root,
        "run_count": run_count,
        "deduplicate_reverse": args.deduplicate_reverse,
    }

    with open(os.path.join(save_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Stage1 swap finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()