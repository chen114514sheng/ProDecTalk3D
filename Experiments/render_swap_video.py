# python Experiments/render_swap_video.py --stage all --pair_type all

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import json
import yaml
import cv2
import imageio
import trimesh
import pyrender
import tempfile
import shutil
import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import ffmpeg
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# 项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DataProcess.Dataload import CustomDataset
from FLAME.FLAME import FLAME


# =========================================================
# 配置
# =========================================================
class SwapVideoConfig:
    SHOW_LABELS = True
    SHOW_HEADER = True

    FONT_PATH = "/usr/share/fonts/dejavu/DejaVuSans.ttf"
    FONT_SIZE = 30
    LABEL_HEIGHT = 50
    HEADER_HEIGHT = 136  # 偶数，避免编码器尺寸问题

    RESIZE_TO = (560, 560)   # (宽, 高)

    # 裁剪参数
    CROP_TOP = 150
    CROP_BOTTOM = 50
    CROP_LEFT = 200
    CROP_RIGHT = 200

    FPS = 25

    # 音频设置
    ADD_AUDIO = True
    AUDIO_DIR = "/mnt/a782f50b-253f-43da-ac99-945477898740/chensheng/Project2/predict/test_audio"

    OUTPUT_ROOT = "Experiments/swap_video"

    # pyrender 离屏渲染尺寸
    RENDER_WIDTH = 960
    RENDER_HEIGHT = 760


# =========================================================
# 基础工具
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_yaml(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npy(path: str) -> np.ndarray:
    return np.load(path)


def ensure_tensor(x, device: torch.device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)


def safe_move(src: str, dst: str):
    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)


def resolve_existing_path(candidates: List[Optional[str]], name: str) -> str:
    for p in candidates:
        if p is not None and os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Cannot find {name}. Tried:\n" + "\n".join([str(x) for x in candidates if x is not None])
    )


def ensure_even_size(frame: np.ndarray) -> np.ndarray:
    """
    libx264 + yuv420p 通常要求宽高为偶数。
    若为奇数，则在右侧/底部补 1 像素白边。
    """
    h, w = frame.shape[:2]
    pad_bottom = 1 if h % 2 != 0 else 0
    pad_right = 1 if w % 2 != 0 else 0

    if pad_bottom == 0 and pad_right == 0:
        return frame

    return cv2.copyMakeBorder(
        frame,
        top=0,
        bottom=pad_bottom,
        left=0,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )


# =========================================================
# 参数解析
# =========================================================
def parse_args():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_config = os.path.join(project_root, "config.yaml")

    parser = argparse.ArgumentParser(description="Render swap experiment videos with GT-inferred valid length.")

    parser.add_argument("--config", type=str, default=default_config)

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "stage1", "stage2"],
        help="Render stage1, stage2, or both"
    )

    parser.add_argument(
        "--pair_type",
        type=str,
        default="all",
        choices=["all", "text_emotion", "text_intensity", "identity"],
        help="Render one swap type or all"
    )

    parser.add_argument(
        "--group_name",
        type=str,
        default=None,
        help="If set, only render this group_xxxxx"
    )

    parser.add_argument(
        "--max_groups",
        type=int,
        default=-1,
        help="Maximum number of groups per setting, -1 means all"
    )

    parser.add_argument("--gpu", type=int, default=None)

    parser.add_argument("--font_path", type=str, default=None)
    parser.add_argument("--font_size", type=int, default=None)

    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--no_audio", action="store_true")

    parser.add_argument(
        "--audio_roles",
        type=str,
        default="both",
        choices=["both", "source", "target"],
        help="Generate source-audio video, target-audio video, or both"
    )

    parser.add_argument("--cell_width", type=int, default=None)
    parser.add_argument("--cell_height", type=int, default=None)

    parser.add_argument("--crop_top", type=int, default=None)
    parser.add_argument("--crop_bottom", type=int, default=None)
    parser.add_argument("--crop_left", type=int, default=None)
    parser.add_argument("--crop_right", type=int, default=None)

    parser.add_argument("--fps", type=int, default=None)

    parser.add_argument("--flame_model_path", type=str, default=None)
    parser.add_argument("--static_landmark_embedding_path", type=str, default=None)
    parser.add_argument("--dynamic_landmark_embedding_path", type=str, default=None)
    parser.add_argument("--template_path", type=str, default=None)

    return parser.parse_args()


# =========================================================
# 数据索引
# =========================================================
def build_dataset_and_index(test_dir: str):
    dataset = CustomDataset(test_dir)
    token_to_index = {}

    for idx, file_path in enumerate(dataset.files):
        token = dataset.extract_video_token(file_path)
        if token not in token_to_index:
            token_to_index[token] = idx

    return dataset, token_to_index


def load_shape_from_token(
    dataset: CustomDataset,
    token_to_index: Dict[str, int],
    video_token: str,
) -> np.ndarray:
    if video_token not in token_to_index:
        raise KeyError(f"video_token not found in dataset: {video_token}")

    idx = token_to_index[video_token]
    _, _, _, _, _, shape_data, _, _, _ = dataset[idx]

    if isinstance(shape_data, torch.Tensor):
        shape_data = shape_data.detach().cpu().numpy()

    shape_data = np.asarray(shape_data, dtype=np.float32)

    if shape_data.ndim == 1:
        shape_data = shape_data[None, :]
    elif shape_data.ndim == 2 and shape_data.shape[0] != 1:
        shape_data = shape_data[:1]

    return shape_data


# =========================================================
# 用 GT 反推 mask / 有效帧长度
# =========================================================
def infer_mask_from_gt(exp_gt: np.ndarray, jaw_gt: np.ndarray) -> np.ndarray:
    """
    根据 GT 反推出有效帧 mask。
    exp_gt: [1, T, D]
    jaw_gt: [1, T, 3]
    返回: [T]，0/1
    """
    if exp_gt.ndim != 3 or jaw_gt.ndim != 3:
        raise ValueError(f"GT shapes should be [1,T,D], got {exp_gt.shape}, {jaw_gt.shape}")

    exp_abs = np.abs(exp_gt).sum(axis=-1)   # [1, T]
    jaw_abs = np.abs(jaw_gt).sum(axis=-1)   # [1, T]
    valid = (exp_abs + jaw_abs) > 0         # [1, T]

    return valid.astype(np.float32)[0]


def mask_to_valid_length(mask: np.ndarray) -> int:
    """
    mask 中 >0 视为有效帧。
    返回最后一个有效帧位置 + 1。
    """
    if mask.ndim != 1:
        mask = mask.reshape(-1)

    valid = mask > 0
    if valid.sum() == 0:
        return len(mask)

    last_valid = np.where(valid)[0][-1]
    return int(last_valid + 1)


def crop_sequence_to_valid_length(exp_arr: np.ndarray, jaw_arr: np.ndarray, valid_len: int):
    """
    exp/jaw 形状为 [1, T, D]
    """
    exp_arr = exp_arr[:, :valid_len, :]
    jaw_arr = jaw_arr[:, :valid_len, :]
    return exp_arr, jaw_arr


# =========================================================
# FLAME 解码
# =========================================================
def build_flame_config(
    flame_model_path: str,
    static_landmark_embedding_path: str,
    dynamic_landmark_embedding_path: str,
    batch_size: int,
    shape_params: int,
    expression_params: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        flame_model_path=flame_model_path,
        static_landmark_embedding_path=static_landmark_embedding_path,
        dynamic_landmark_embedding_path=dynamic_landmark_embedding_path,
        batch_size=batch_size,
        use_face_contour=False,
        shape_params=shape_params,
        expression_params=expression_params,
        use_3D_translation=False,
    )


class FlameDecoderCache:
    def __init__(
        self,
        flame_model_path: str,
        static_landmark_embedding_path: str,
        dynamic_landmark_embedding_path: str,
        device: torch.device
    ):
        self.flame_model_path = flame_model_path
        self.static_landmark_embedding_path = static_landmark_embedding_path
        self.dynamic_landmark_embedding_path = dynamic_landmark_embedding_path
        self.device = device
        self.cache = {}

    def get(self, shape_dim: int, exp_dim: int):
        key = (shape_dim, exp_dim)
        if key in self.cache:
            return self.cache[key]

        cfg = build_flame_config(
            flame_model_path=self.flame_model_path,
            static_landmark_embedding_path=self.static_landmark_embedding_path,
            dynamic_landmark_embedding_path=self.dynamic_landmark_embedding_path,
            batch_size=1,
            shape_params=int(shape_dim),
            expression_params=int(exp_dim),
        )

        flame = FLAME(cfg).to(self.device)
        flame.eval()
        self.cache[key] = flame
        return flame


def decode_vertices_sequence(
    shape_arr: np.ndarray,   # [1, Ds]
    exp_arr: np.ndarray,     # [1, T, De]
    jaw_arr: np.ndarray,     # [1, T, 3]
    flame_cache: FlameDecoderCache,
    device: torch.device,
) -> np.ndarray:
    shape_arr = np.asarray(shape_arr, dtype=np.float32)
    exp_arr = np.asarray(exp_arr, dtype=np.float32)
    jaw_arr = np.asarray(jaw_arr, dtype=np.float32)

    if shape_arr.ndim != 2:
        raise ValueError(f"shape_arr should be [1, D], got {shape_arr.shape}")
    if exp_arr.ndim != 3 or jaw_arr.ndim != 3:
        raise ValueError(f"exp/jaw should be [1, T, D], got {exp_arr.shape}, {jaw_arr.shape}")

    T = exp_arr.shape[1]
    shape_dim = shape_arr.shape[-1]
    exp_dim = exp_arr.shape[-1]

    flame = flame_cache.get(shape_dim, exp_dim)

    vertices_list = []
    with torch.no_grad():
        shape_t = ensure_tensor(shape_arr, device)

        for t in range(T):
            exp_t = ensure_tensor(exp_arr[:, t, :], device)
            jaw_t = ensure_tensor(jaw_arr[:, t, :], device)

            pose_t = torch.zeros((1, 6), dtype=torch.float32, device=device)
            pose_t[:, 3:] = jaw_t

            vertices, _ = flame(
                shape_params=shape_t,
                expression_params=exp_t,
                pose_params=pose_t,
            )
            vertices_list.append(vertices[0].detach().cpu().numpy())

    return np.stack(vertices_list, axis=0)


# =========================================================
# 渲染
# =========================================================
class MeshRenderer:
    def __init__(self, template_path: str):
        self.template_mesh = trimesh.load_mesh(template_path)
        self.faces = self.template_mesh.faces.copy()

        self.cam = pyrender.PerspectiveCamera(yfov=np.pi / 20, aspectRatio=1.414)
        self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
        self.camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.renderer = pyrender.OffscreenRenderer(
            SwapVideoConfig.RENDER_WIDTH,
            SwapVideoConfig.RENDER_HEIGHT
        )

    def render_frame(self, vertices: np.ndarray) -> np.ndarray:
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        py_mesh = pyrender.Mesh.from_trimesh(mesh)

        scene = pyrender.Scene()
        scene.add(py_mesh)
        scene.add(self.cam, pose=self.camera_pose)
        scene.add(self.light, pose=self.camera_pose)

        color, _ = self.renderer.render(scene)
        return color

    def close(self):
        self.renderer.delete()


# =========================================================
# 排版
# =========================================================
def create_label_image(label: str, width: int, height: int) -> np.ndarray:
    img = Image.new('RGB', (width, height), (0, 0, 0))
    try:
        font = ImageFont.truetype(SwapVideoConfig.FONT_PATH, SwapVideoConfig.FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        ((width - text_w) // 2, (height - text_h) // 2),
        label,
        fill="white",
        font=font
    )
    return np.array(img)


def create_header_image(text_lines: List[str], width: int, height: int) -> np.ndarray:
    img = Image.new('RGB', (width, height), (255, 255, 255))
    try:
        font = ImageFont.truetype(SwapVideoConfig.FONT_PATH, 24)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    y = 10
    for line in text_lines:
        draw.text((12, y), line, fill="black", font=font)
        y += 24
    return np.array(img)


def crop_frame(frame: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    h, w = frame.shape[:2]
    new_h = h - top - bottom
    new_w = w - left - right
    if new_h <= 0 or new_w <= 0:
        return frame
    return frame[top:h-bottom, left:w-right]


def prepare_panel(frame: np.ndarray, label: str) -> np.ndarray:
    frame = crop_frame(
        frame,
        SwapVideoConfig.CROP_TOP,
        SwapVideoConfig.CROP_BOTTOM,
        SwapVideoConfig.CROP_LEFT,
        SwapVideoConfig.CROP_RIGHT
    )

    target_w, target_h = SwapVideoConfig.RESIZE_TO
    if frame.shape[1] != target_w or frame.shape[0] != target_h:
        frame = cv2.resize(frame, (target_w, target_h))

    if SwapVideoConfig.SHOW_LABELS:
        label_img = create_label_image(label, target_w, SwapVideoConfig.LABEL_HEIGHT)
        frame = np.vstack([label_img, frame])

    return frame


def compose_two_panel_layout(left_frame: np.ndarray, right_frame: np.ndarray, header: Optional[np.ndarray] = None) -> np.ndarray:
    frame_h, frame_w = left_frame.shape[:2]
    canvas = np.zeros((frame_h, frame_w * 2, 3), dtype=np.uint8)
    canvas[:, 0:frame_w] = left_frame
    canvas[:, frame_w:2 * frame_w] = right_frame

    if header is not None:
        canvas = np.vstack([header, canvas])

    return ensure_even_size(canvas)


# =========================================================
# 音频：ffmpeg-python
# =========================================================
def extract_audio(input_path: str, audio_out_path: str, duration: float) -> bool:
    try:
        (
            ffmpeg
            .input(input_path, t=duration)
            .output(audio_out_path, acodec='aac', vn=None)
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except Exception:
        return False


def add_audio_to_video(video_path: str, audio_path: str, output_path: str) -> bool:
    try:
        video_in = ffmpeg.input(video_path)
        audio_in = ffmpeg.input(audio_path)

        (
            ffmpeg
            .output(
                video_in.video,
                audio_in.audio,
                output_path,
                vcodec='copy',
                acodec='aac',
                shortest=None
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except Exception:
        return False


def get_audio_path_from_meta(meta: Dict[str, Any], audio_role: str) -> Optional[str]:
    token = meta["source_video_token"] if audio_role == "source" else meta["target_video_token"]
    wav_path = os.path.join(SwapVideoConfig.AUDIO_DIR, f"{token}.wav")
    if os.path.exists(wav_path):
        return wav_path
    return None


# =========================================================
# 结果目录与键
# =========================================================
def get_result_root(stage: str, pair_type: str) -> str:
    if stage == "stage1":
        return os.path.join("Experiments", "stage1_swap_results", pair_type)
    elif stage == "stage2":
        return os.path.join("Experiments", "stage2_swap_results", pair_type)
    else:
        raise ValueError(f"Unsupported stage: {stage}")


def get_stage_result_keys(stage: str) -> Dict[str, str]:
    if stage == "stage1":
        return {
            "source_orig_exp": "source_rec_exp.npy",
            "source_orig_jaw": "source_rec_jaw.npy",
            "target_orig_exp": "target_rec_exp.npy",
            "target_orig_jaw": "target_rec_jaw.npy",
        }
    elif stage == "stage2":
        return {
            "source_orig_exp": "source_gen_exp.npy",
            "source_orig_jaw": "source_gen_jaw.npy",
            "target_orig_exp": "target_gen_exp.npy",
            "target_orig_jaw": "target_gen_jaw.npy",
        }
    else:
        raise ValueError(f"Unsupported stage: {stage}")


def list_group_dirs(root_dir: str) -> List[str]:
    if not os.path.exists(root_dir):
        return []
    out = []
    for name in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and name.startswith("group_"):
            out.append(full)
    return out


# =========================================================
# 核心：渲染单个 group 的一个音频版本（基于 GT 推断有效帧）
# =========================================================
def render_group_video(
    stage: str,
    pair_type: str,
    group_dir: str,
    audio_role: str,
    dataset: CustomDataset,
    token_to_index: Dict[str, int],
    flame_cache: FlameDecoderCache,
    mesh_renderer: MeshRenderer,
    save_dir: str,
):
    meta = load_json(os.path.join(group_dir, "meta.json"))
    keys = get_stage_result_keys(stage)

    source_token = meta["source_video_token"]
    target_token = meta["target_video_token"]

    source_shape = load_shape_from_token(dataset, token_to_index, source_token)
    target_shape = load_shape_from_token(dataset, token_to_index, target_token)

    source_gt_exp = load_npy(os.path.join(group_dir, "source_gt_exp.npy"))
    source_gt_jaw = load_npy(os.path.join(group_dir, "source_gt_jaw.npy"))
    target_gt_exp = load_npy(os.path.join(group_dir, "target_gt_exp.npy"))
    target_gt_jaw = load_npy(os.path.join(group_dir, "target_gt_jaw.npy"))

    source_orig_exp = load_npy(os.path.join(group_dir, keys["source_orig_exp"]))
    source_orig_jaw = load_npy(os.path.join(group_dir, keys["source_orig_jaw"]))
    target_orig_exp = load_npy(os.path.join(group_dir, keys["target_orig_exp"]))
    target_orig_jaw = load_npy(os.path.join(group_dir, keys["target_orig_jaw"]))

    source_swap_exp = load_npy(os.path.join(group_dir, "source_swap_exp.npy"))
    source_swap_jaw = load_npy(os.path.join(group_dir, "source_swap_jaw.npy"))
    target_swap_exp = load_npy(os.path.join(group_dir, "target_swap_exp.npy"))
    target_swap_jaw = load_npy(os.path.join(group_dir, "target_swap_jaw.npy"))

    device = flame_cache.device

    if audio_role == "source":
        gt_mask = infer_mask_from_gt(source_gt_exp, source_gt_jaw)
        raw_padded_len = int(source_gt_exp.shape[1])
        valid_len = mask_to_valid_length(gt_mask)

        orig_shape = source_shape
        swap_shape = source_shape

        orig_exp, orig_jaw = crop_sequence_to_valid_length(source_orig_exp, source_orig_jaw, valid_len)
        swap_exp, swap_jaw = crop_sequence_to_valid_length(source_swap_exp, source_swap_jaw, valid_len)

        token_main = source_token
        token_other = target_token
        left_label = "Source Original"
        right_label = "Source Swapped"

    elif audio_role == "target":
        gt_mask = infer_mask_from_gt(target_gt_exp, target_gt_jaw)
        raw_padded_len = int(target_gt_exp.shape[1])
        valid_len = mask_to_valid_length(gt_mask)

        orig_shape = target_shape
        swap_shape = target_shape

        orig_exp, orig_jaw = crop_sequence_to_valid_length(target_orig_exp, target_orig_jaw, valid_len)
        swap_exp, swap_jaw = crop_sequence_to_valid_length(target_swap_exp, target_swap_jaw, valid_len)

        token_main = target_token
        token_other = source_token
        left_label = "Target Original"
        right_label = "Target Swapped"
    else:
        raise ValueError(f"Unsupported audio_role: {audio_role}")

    orig_vertices = decode_vertices_sequence(orig_shape, orig_exp, orig_jaw, flame_cache, device)
    swap_vertices = decode_vertices_sequence(swap_shape, swap_exp, swap_jaw, flame_cache, device)

    T = min(len(orig_vertices), len(swap_vertices), valid_len)

    ensure_dir(save_dir)
    group_name = Path(group_dir).name
    output_path = os.path.join(save_dir, f"{group_name}_audio_{audio_role}.mp4")

    tmp_video = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
        dir=save_dir
    ).name

    writer = imageio.get_writer(
        tmp_video,
        fps=SwapVideoConfig.FPS,
        macro_block_size=1,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p'
    )

    header = None
    if SwapVideoConfig.SHOW_HEADER:
        header_text = [
            f"{stage} | {pair_type} | {group_name}",
            f"Main Token: {token_main}",
            f"Reference Token: {token_other}",
            f"Audio Source: {audio_role}",
            f"Valid Frames: {valid_len} / {raw_padded_len}",
        ]
        panel_w = SwapVideoConfig.RESIZE_TO[0]
        total_w = panel_w * 2
        header = create_header_image(header_text, total_w, SwapVideoConfig.HEADER_HEIGHT)

    try:
        for t in tqdm(range(T), desc=f"Rendering {group_name} [{audio_role}]"):
            fr_left = mesh_renderer.render_frame(orig_vertices[t])
            fr_right = mesh_renderer.render_frame(swap_vertices[t])

            p_left = prepare_panel(fr_left, left_label)
            p_right = prepare_panel(fr_right, right_label)

            composed = compose_two_panel_layout(p_left, p_right, header=header)
            writer.append_data(composed)

        writer.close()

        if SwapVideoConfig.ADD_AUDIO:
            audio_path = get_audio_path_from_meta(meta, audio_role)
            if audio_path is not None:
                tmp_audio = tempfile.NamedTemporaryFile(
                    suffix=".aac",
                    delete=False,
                    dir=save_dir
                ).name

                duration = T / float(SwapVideoConfig.FPS)
                ok1 = extract_audio(audio_path, tmp_audio, duration)
                ok2 = add_audio_to_video(tmp_video, tmp_audio, output_path) if ok1 else False

                if ok2:
                    print(f"✅ Saved with {audio_role} audio: {output_path}")
                    try:
                        os.remove(tmp_video)
                    except Exception:
                        pass
                    try:
                        os.remove(tmp_audio)
                    except Exception:
                        pass
                else:
                    safe_move(tmp_video, output_path)
                    print(f"⚠️ Audio mux failed, saved video only: {output_path}")
                    try:
                        if os.path.exists(tmp_audio):
                            os.remove(tmp_audio)
                    except Exception:
                        pass
            else:
                safe_move(tmp_video, output_path)
                print(f"⚠️ {audio_role} audio not found, saved video only: {output_path}")
        else:
            safe_move(tmp_video, output_path)
            print(f"✅ Saved: {output_path}")

    except Exception as e:
        writer.close()
        try:
            if os.path.exists(tmp_video):
                os.remove(tmp_video)
        except Exception:
            pass
        raise e


# =========================================================
# 主流程
# =========================================================
def main():
    args = parse_args()
    config = load_yaml(args.config)

    if args.font_path is not None:
        SwapVideoConfig.FONT_PATH = args.font_path
    if args.font_size is not None:
        SwapVideoConfig.FONT_SIZE = args.font_size
    if args.audio_dir is not None:
        SwapVideoConfig.AUDIO_DIR = args.audio_dir
    if args.no_audio:
        SwapVideoConfig.ADD_AUDIO = False
    if args.cell_width is not None:
        SwapVideoConfig.RESIZE_TO = (args.cell_width, SwapVideoConfig.RESIZE_TO[1])
    if args.cell_height is not None:
        SwapVideoConfig.RESIZE_TO = (SwapVideoConfig.RESIZE_TO[0], args.cell_height)
    if args.crop_top is not None:
        SwapVideoConfig.CROP_TOP = args.crop_top
    if args.crop_bottom is not None:
        SwapVideoConfig.CROP_BOTTOM = args.crop_bottom
    if args.crop_left is not None:
        SwapVideoConfig.CROP_LEFT = args.crop_left
    if args.crop_right is not None:
        SwapVideoConfig.CROP_RIGHT = args.crop_right
    if args.fps is not None:
        SwapVideoConfig.FPS = args.fps

    if args.audio_roles == "both":
        audio_roles = ["source", "target"]
    else:
        audio_roles = [args.audio_roles]

    device = torch.device(
        f"cuda:{args.gpu if args.gpu is not None else config['predict']['gpu']}"
        if torch.cuda.is_available() else "cpu"
    )

    flame_model_path = resolve_existing_path([
        args.flame_model_path,
        "/home/chensheng/1Project/Project2/FLAME/flame_model/generic_model.pkl",
        "/home/chensheng/1Project/Project2/FLAME/generic_model.pkl",
    ], "flame_model_path")

    static_landmark_embedding_path = resolve_existing_path([
        args.static_landmark_embedding_path,
        "/home/chensheng/1Project/Project2/FLAME/flame_model/flame_static_embedding.pkl",
        "/home/chensheng/1Project/Project2/FLAME/flame_model/static_landmark_embedding.pkl",
        "/home/chensheng/1Project/Project2/FLAME/flame_model/static_embedding.pkl",
        "/home/chensheng/1Project/Project2/FLAME/flame_static_embedding.pkl",
    ], "static_landmark_embedding_path")

    dynamic_landmark_embedding_path = resolve_existing_path([
        args.dynamic_landmark_embedding_path,
        "/home/chensheng/1Project/Project2/FLAME/flame_model/flame_dynamic_embedding.npy",
        "/home/chensheng/1Project/Project2/FLAME/flame_model/dynamic_landmark_embedding.npy",
        "/home/chensheng/1Project/Project2/FLAME/flame_model/dynamic_embedding.npy",
        "/home/chensheng/1Project/Project2/FLAME/flame_dynamic_embedding.npy",
    ], "dynamic_landmark_embedding_path")

    template_path = resolve_existing_path([
        args.template_path,
        "/home/chensheng/1Project/Project2/FLAME/flame_sample.ply",
    ], "template_path")

    test_dir = config["test_file_path"]
    dataset, token_to_index = build_dataset_and_index(test_dir)

    flame_cache = FlameDecoderCache(
        flame_model_path=flame_model_path,
        static_landmark_embedding_path=static_landmark_embedding_path,
        dynamic_landmark_embedding_path=dynamic_landmark_embedding_path,
        device=device
    )
    mesh_renderer = MeshRenderer(template_path)

    if args.stage == "all":
        stages = ["stage1", "stage2"]
    else:
        stages = [args.stage]

    if args.pair_type == "all":
        pair_types = ["text_emotion", "text_intensity", "identity"]
    else:
        pair_types = [args.pair_type]

    print("=" * 80)
    print(f"stages      : {stages}")
    print(f"pair_types  : {pair_types}")
    print(f"audio_roles : {audio_roles}")
    print(f"audio       : {SwapVideoConfig.ADD_AUDIO}")
    print(f"font        : {SwapVideoConfig.FONT_PATH}")
    print(f"output_root : {SwapVideoConfig.OUTPUT_ROOT}")
    print("layout      : horizontal two-panel")
    print("mask        : inferred from GT")
    print("=" * 80)

    try:
        for stage in stages:
            for pair_type in pair_types:
                result_root = get_result_root(stage, pair_type)
                save_dir = os.path.join(SwapVideoConfig.OUTPUT_ROOT, stage, pair_type)
                ensure_dir(save_dir)

                if args.group_name is not None:
                    group_dirs = [os.path.join(result_root, args.group_name)]
                else:
                    group_dirs = list_group_dirs(result_root)
                    if args.max_groups > 0:
                        group_dirs = group_dirs[:args.max_groups]

                print("-" * 80)
                print(f"stage       : {stage}")
                print(f"pair_type   : {pair_type}")
                print(f"result_root : {result_root}")
                print(f"save_dir    : {save_dir}")
                print(f"num_groups  : {len(group_dirs)}")
                print("-" * 80)

                for group_dir in group_dirs:
                    if not os.path.exists(group_dir):
                        print(f"[Skip] group dir does not exist: {group_dir}")
                        continue

                    for audio_role in audio_roles:
                        try:
                            render_group_video(
                                stage=stage,
                                pair_type=pair_type,
                                group_dir=group_dir,
                                audio_role=audio_role,
                                dataset=dataset,
                                token_to_index=token_to_index,
                                flame_cache=flame_cache,
                                mesh_renderer=mesh_renderer,
                                save_dir=save_dir,
                            )
                        except Exception as e:
                            print(f"[Skip] {group_dir} [{audio_role}] because of error: {e}")
    finally:
        mesh_renderer.close()

    print("=" * 80)
    print("Swap video rendering finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()