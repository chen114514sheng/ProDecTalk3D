# Experiments/build_swap_pairs.py

import os
import sys
import json
import yaml
import random
import argparse
from collections import defaultdict
from typing import Dict, List

# 项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DataProcess.Dataload import CustomDataset


def load_yaml(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)


def parse_args():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_config = os.path.join(project_root, "config.yaml")
    default_output_dir = os.path.join(os.path.dirname(__file__), "swap_pairs")

    parser = argparse.ArgumentParser(description="Build swap pairs for condition swapping experiments.")
    parser.add_argument("--config", type=str, default=default_config, help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Override test dataset path")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Directory to save pair json files")
    parser.add_argument(
        "--pair_type",
        type=str,
        default="all",
        choices=["all", "text_emotion", "text_intensity", "identity"],
        help="Which pair type to build"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max_pairs_per_group",
        type=int,
        default=20,
        help="Maximum number of bidirectional pairs sampled from each group"
    )
    return parser.parse_args()


def parse_video_token(video_token: str) -> Dict:
    """
    根据你当前 Dataload.py 的逻辑：
    parts[0] -> person_id
    parts[2] -> emotion
    parts[3] -> intensity

    其余部分作为 sentence_key / utterance_key 使用。
    """
    parts = video_token.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected video_token format: {video_token}")

    person_id = parts[0]
    clip_or_misc = parts[1]
    emotion = parts[2]
    intensity = parts[3]
    sentence_key = "_".join(parts[4:])

    return {
        "video_token": video_token,
        "person_id": person_id,
        "clip_or_misc": clip_or_misc,
        "emotion": emotion,
        "intensity": intensity,
        "sentence_key": sentence_key,
    }


def build_sample_records(dataset: CustomDataset) -> List[Dict]:
    """
    只读取文件路径和 token，不加载 h5 数据，速度更快。
    """
    records = []
    for file_path in dataset.files:
        video_token = dataset.extract_video_token(file_path)
        meta = parse_video_token(video_token)
        meta["file_path"] = file_path
        records.append(meta)
    return records


def sample_bidirectional_pairs(items: List[Dict], max_pairs_per_group: int) -> List[Dict]:
    """
    给定一个组内样本列表，生成若干双向配对。
    例如 items=[A,B,C]，可采样 (A,B), (B,A), (A,C), (C,A) ...
    """
    if len(items) < 2:
        return []

    all_pairs = []
    n = len(items)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            all_pairs.append((items[i], items[j]))

    random.shuffle(all_pairs)
    all_pairs = all_pairs[:max_pairs_per_group]

    results = []
    for a, b in all_pairs:
        results.append({
            "source_video_token": a["video_token"],
            "target_video_token": b["video_token"],
            "source_file_path": a["file_path"],
            "target_file_path": b["file_path"],

            "source_person_id": a["person_id"],
            "target_person_id": b["person_id"],

            "source_emotion": a["emotion"],
            "target_emotion": b["emotion"],

            "source_intensity": a["intensity"],
            "target_intensity": b["intensity"],

            "source_sentence_key": a["sentence_key"],
            "target_sentence_key": b["sentence_key"],
        })
    return results


def build_text_emotion_pairs(records: List[Dict], max_pairs_per_group: int) -> List[Dict]:
    """
    文本情感类别交换：
    - 同一身份
    - 同一句子
    - 同一强度
    - 不同情感类别
    """
    grouped = defaultdict(list)
    for r in records:
        key = (r["person_id"], r["sentence_key"], r["intensity"])
        grouped[key].append(r)

    pair_list = []
    for _, items in grouped.items():
        # 组内先按 emotion 再细分
        by_emotion = defaultdict(list)
        for item in items:
            by_emotion[item["emotion"]].append(item)

        emotions = list(by_emotion.keys())
        if len(emotions) < 2:
            continue

        candidates = []
        for emo_a in emotions:
            for emo_b in emotions:
                if emo_a == emo_b:
                    continue
                for a in by_emotion[emo_a]:
                    for b in by_emotion[emo_b]:
                        candidates.append((a, b))

        random.shuffle(candidates)
        candidates = candidates[:max_pairs_per_group]

        for a, b in candidates:
            pair_list.append({
                "pair_type": "text_emotion",
                "source_video_token": a["video_token"],
                "target_video_token": b["video_token"],
                "source_file_path": a["file_path"],
                "target_file_path": b["file_path"],

                "person_id": a["person_id"],
                "sentence_key": a["sentence_key"],

                "source_emotion": a["emotion"],
                "target_emotion": b["emotion"],

                "source_intensity": a["intensity"],
                "target_intensity": b["intensity"],  # 应当相同
            })

    return pair_list


def build_text_intensity_pairs(records: List[Dict], max_pairs_per_group: int) -> List[Dict]:
    """
    文本情感强度交换：
    - 同一身份
    - 同一句子
    - 同一情感类别
    - 不同强度
    """
    grouped = defaultdict(list)
    for r in records:
        key = (r["person_id"], r["sentence_key"], r["emotion"])
        grouped[key].append(r)

    pair_list = []
    for _, items in grouped.items():
        by_intensity = defaultdict(list)
        for item in items:
            by_intensity[item["intensity"]].append(item)

        intensities = list(by_intensity.keys())
        if len(intensities) < 2:
            continue

        candidates = []
        for inten_a in intensities:
            for inten_b in intensities:
                if inten_a == inten_b:
                    continue
                for a in by_intensity[inten_a]:
                    for b in by_intensity[inten_b]:
                        candidates.append((a, b))

        random.shuffle(candidates)
        candidates = candidates[:max_pairs_per_group]

        for a, b in candidates:
            pair_list.append({
                "pair_type": "text_intensity",
                "source_video_token": a["video_token"],
                "target_video_token": b["video_token"],
                "source_file_path": a["file_path"],
                "target_file_path": b["file_path"],

                "person_id": a["person_id"],
                "sentence_key": a["sentence_key"],

                "source_emotion": a["emotion"],
                "target_emotion": b["emotion"],  # 应当相同

                "source_intensity": a["intensity"],
                "target_intensity": b["intensity"],
            })

    return pair_list


def build_identity_pairs(records: List[Dict], max_pairs_per_group: int) -> List[Dict]:
    """
    身份交换：
    - 同一句子
    - 同一情感类别
    - 同一强度
    - 不同身份
    """
    grouped = defaultdict(list)
    for r in records:
        key = (r["sentence_key"], r["emotion"], r["intensity"])
        grouped[key].append(r)

    pair_list = []
    for _, items in grouped.items():
        by_person = defaultdict(list)
        for item in items:
            by_person[item["person_id"]].append(item)

        persons = list(by_person.keys())
        if len(persons) < 2:
            continue

        candidates = []
        for pid_a in persons:
            for pid_b in persons:
                if pid_a == pid_b:
                    continue
                for a in by_person[pid_a]:
                    for b in by_person[pid_b]:
                        candidates.append((a, b))

        random.shuffle(candidates)
        candidates = candidates[:max_pairs_per_group]

        for a, b in candidates:
            pair_list.append({
                "pair_type": "identity",
                "source_video_token": a["video_token"],
                "target_video_token": b["video_token"],
                "source_file_path": a["file_path"],
                "target_file_path": b["file_path"],

                "source_person_id": a["person_id"],
                "target_person_id": b["person_id"],

                "sentence_key": a["sentence_key"],
                "emotion": a["emotion"],
                "intensity": a["intensity"],
            })

    return pair_list


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_stats(name: str, pairs: List[Dict]):
    print("=" * 80)
    print(f"{name}: {len(pairs)} pairs")
    if len(pairs) > 0:
        print("Example:")
        print(json.dumps(pairs[0], indent=2, ensure_ascii=False))
    print("=" * 80)


def main():
    args = parse_args()
    set_seed(args.seed)

    config = load_yaml(args.config)
    data_dir = args.data_dir if args.data_dir is not None else config["test_file_path"]
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = CustomDataset(data_dir)
    records = build_sample_records(dataset)

    print(f"Loaded {len(records)} records from: {data_dir}")

    if args.pair_type in ["all", "text_emotion"]:
        text_emotion_pairs = build_text_emotion_pairs(records, args.max_pairs_per_group)
        save_json(os.path.join(args.output_dir, "text_emotion_pairs.json"), text_emotion_pairs)
        print_stats("text_emotion_pairs", text_emotion_pairs)

    if args.pair_type in ["all", "text_intensity"]:
        text_intensity_pairs = build_text_intensity_pairs(records, args.max_pairs_per_group)
        save_json(os.path.join(args.output_dir, "text_intensity_pairs.json"), text_intensity_pairs)
        print_stats("text_intensity_pairs", text_intensity_pairs)

    if args.pair_type in ["all", "identity"]:
        identity_pairs = build_identity_pairs(records, args.max_pairs_per_group)
        save_json(os.path.join(args.output_dir, "identity_pairs.json"), identity_pairs)
        print_stats("identity_pairs", identity_pairs)


if __name__ == "__main__":
    main()