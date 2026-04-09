# Experiments/run_stage2_swap.py

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
from Diffusion.Diffusion import FaceGenerationModel


def load_yaml(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_config = os.path.join(project_root, "config.yaml")
    default_pair_dir = os.path.join(os.path.dirname(__file__), "swap_pairs")
    default_save_root = os.path.join(os.path.dirname(__file__), "stage2_swap_results")

    parser = argparse.ArgumentParser(description="Run stage2 swap experiments for diffusion generation stage.")
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
        help="Root directory to save stage2 swap outputs"
    )
    parser.add_argument("--gpu", type=int, default=None, help="Override gpu id")
    parser.add_argument("--max_groups", type=int, default=-1, help="Maximum number of pair groups to run, -1 means all")
    parser.add_argument(
        "--deduplicate_reverse",
        action="store_true",
        help="Skip reverse-duplicate pairs, only keep one group for {A,B}"
    )

    # 可覆盖采样参数；若不传则优先走 config.predict，再退回到这里的默认值
    parser.add_argument("--num_sampling_steps_top", type=int, default=None)
    parser.add_argument("--num_sampling_steps_bottom", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)

    return parser.parse_args()


def get_stage2_ckpt_path(config: Dict) -> str:
    """
    优先使用 predict.diffusion_dir；
    若不存在，再回退到 stage2.checkpoint_dir/model_val.pth
    """
    ckpt_path = config.get("predict", {}).get("diffusion_dir", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        return ckpt_path

    stage2_ckpt_dir = config["stage2"]["checkpoint_dir"]
    fallback = os.path.join(stage2_ckpt_dir, "model_val.pth")
    if os.path.exists(fallback):
        return fallback

    raise FileNotFoundError(
        f"Cannot find stage2 checkpoint. "
        f"Tried predict.vq_dir={ckpt_path} and fallback={fallback}"
    )


def get_stage1_ckpt_for_stage2(config: Dict) -> str:
    """
    第二阶段模型内部要加载第一阶段 VQVAE 权重。
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
        f"Cannot find stage1 checkpoint for stage2 model. "
        f"Tried predict.vqvae2_dir={ckpt_path} and fallback={fallback}"
    )


def get_predict_sampling_args(config: Dict, args) -> Dict[str, Any]:
    predict_cfg = config.get("predict", {})

    num_sampling_steps_top = (
        args.num_sampling_steps_top
        if args.num_sampling_steps_top is not None
        else predict_cfg.get("num_sampling_steps_top", 25)
    )
    num_sampling_steps_bottom = (
        args.num_sampling_steps_bottom
        if args.num_sampling_steps_bottom is not None
        else predict_cfg.get("num_sampling_steps_bottom", 25)
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else predict_cfg.get("temperature", 0.2)
    )
    top_k = (
        args.top_k
        if args.top_k is not None
        else predict_cfg.get("k", 5)
    )

    return {
        "num_sampling_steps_top": num_sampling_steps_top,
        "num_sampling_steps_bottom": num_sampling_steps_bottom,
        "temperature": temperature,
        "k": top_k,
    }


def build_model(config: Dict, device: torch.device) -> FaceGenerationModel:
    gpu_id = device.index if device.type == "cuda" else 0

    model = FaceGenerationModel(
        vqvae_dir=get_stage1_ckpt_for_stage2(config),
        embed_dim=config["stage1"]["embed_dim"],
        num_heads1=config["stage1"]["num_heads"],
        num_layers_style=config["stage1"]["num_layers_style"],
        num_layers_top1=config["stage1"]["num_layers_top"],
        num_layers_bottom1=config["stage1"]["num_layers_bottom"],
        num_embeddings=config["stage1"]["num_embeddings"],
        num_heads2=config["stage2"]["num_heads"],
        num_layers_temporal=config["stage2"]["num_layers_temporal"],
        num_layers_semantic=config["stage2"]["num_layers_semantic"],
        num_layers=config["stage2"]["num_layers"],
        gpu=gpu_id,
        num_diffusion_timesteps=config["stage2"].get("num_diffusion_timesteps", 1000),
        temperature=config["stage2"].get("temperature", 1.0),
    ).to(device)

    ckpt_path = get_stage2_ckpt_path(config)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    print("=" * 80)
    print("Loaded stage2 model checkpoint:")
    print(ckpt_path)
    print("Loaded stage1 VQVAE checkpoint for stage2:")
    print(get_stage1_ckpt_for_stage2(config))
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
        "audio": ensure_tensor(audio, device).unsqueeze(0) if not isinstance(audio, torch.Tensor) or audio.dim() == 2 else ensure_tensor(audio, device),
        "shape_data": ensure_tensor(shape_data, device).unsqueeze(0),
        "exp_data": ensure_tensor(exp_data, device).unsqueeze(0),
        "jaw_data": ensure_tensor(jaw_data, device).unsqueeze(0),
        "mask": ensure_tensor(mask, device).unsqueeze(0),
    }
    return sample


@torch.no_grad()
def generate_original(
    model: FaceGenerationModel,
    sample: Dict[str, Any],
    sampling_args: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    exp_out, jaw_out = model.sample(
        person_one_hot=sample["person_one_hot"],
        text=sample["text"],
        audio=sample["audio"],
        num_sampling_steps_top=sampling_args["num_sampling_steps_top"],
        num_sampling_steps_bottom=sampling_args["num_sampling_steps_bottom"],
        temperature=sampling_args["temperature"],
        k=sampling_args["k"],
    )
    return exp_out, jaw_out


@torch.no_grad()
def generate_swapped(
    model: FaceGenerationModel,
    source_sample: Dict[str, Any],
    target_sample: Dict[str, Any],
    pair_type: str,
    sampling_args: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    交换原则：
    1) text_emotion / text_intensity: 固定源 audio 和源 identity，只换目标文本
    2) identity: 固定源 audio 和源 text，只换目标身份
    """
    decode_person = source_sample["person_one_hot"]
    decode_text = source_sample["text"]
    decode_audio = source_sample["audio"]

    if pair_type in ["text_emotion", "text_intensity"]:
        decode_text = target_sample["text"]
    elif pair_type == "identity":
        decode_person = target_sample["person_one_hot"]
    else:
        raise ValueError(f"Unsupported pair_type: {pair_type}")

    exp_out, jaw_out = model.sample(
        person_one_hot=decode_person,
        text=decode_text,
        audio=decode_audio,
        num_sampling_steps_top=sampling_args["num_sampling_steps_top"],
        num_sampling_steps_bottom=sampling_args["num_sampling_steps_bottom"],
        temperature=sampling_args["temperature"],
        k=sampling_args["k"],
    )
    return exp_out, jaw_out


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
    source_gen: Tuple[torch.Tensor, torch.Tensor],
    target_gen: Tuple[torch.Tensor, torch.Tensor],
    source_swap: Tuple[torch.Tensor, torch.Tensor],
    target_swap: Tuple[torch.Tensor, torch.Tensor],
):
    os.makedirs(save_dir, exist_ok=True)

    # 保存 GT
    save_array(os.path.join(save_dir, "source_gt_exp.npy"), tensor_to_numpy(source_sample["exp_data"]))
    save_array(os.path.join(save_dir, "source_gt_jaw.npy"), tensor_to_numpy(source_sample["jaw_data"]))
    save_array(os.path.join(save_dir, "target_gt_exp.npy"), tensor_to_numpy(target_sample["exp_data"]))
    save_array(os.path.join(save_dir, "target_gt_jaw.npy"), tensor_to_numpy(target_sample["jaw_data"]))

    # 保存原生成
    save_array(os.path.join(save_dir, "source_gen_exp.npy"), tensor_to_numpy(source_gen[0]))
    save_array(os.path.join(save_dir, "source_gen_jaw.npy"), tensor_to_numpy(source_gen[1]))
    save_array(os.path.join(save_dir, "target_gen_exp.npy"), tensor_to_numpy(target_gen[0]))
    save_array(os.path.join(save_dir, "target_gen_jaw.npy"), tensor_to_numpy(target_gen[1]))

    # 保存交换生成
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
            "source_gen_exp.npy", "source_gen_jaw.npy",
            "target_gen_exp.npy", "target_gen_jaw.npy",
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
    sampling_args = get_predict_sampling_args(config, args)

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

        # 原生成
        source_gen = generate_original(model, source_sample, sampling_args)
        target_gen = generate_original(model, target_sample, sampling_args)

        # 双向交换生成
        # source_swap: A <- B
        source_swap = generate_swapped(model, source_sample, target_sample, args.pair_type, sampling_args)
        # target_swap: B <- A
        target_swap = generate_swapped(model, target_sample, source_sample, args.pair_type, sampling_args)

        group_name = f"group_{run_count:05d}"
        group_save_dir = os.path.join(save_root, group_name)

        save_group_results(
            save_dir=group_save_dir,
            pair_meta=pair,
            source_sample=source_sample,
            target_sample=target_sample,
            source_gen=source_gen,
            target_gen=target_gen,
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
        "sampling_args": sampling_args,
    }

    with open(os.path.join(save_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Stage2 swap finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()