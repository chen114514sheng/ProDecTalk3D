# Experiments/render_swap_vis.py
# python Experiments/render_swap_vis.py --stage all --pair_type all

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import json
import yaml
import argparse
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import cv2
import numpy as np
import torch
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont

# 项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DataProcess.Dataload import CustomDataset
from FLAME.FLAME import FLAME


def load_yaml(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_config = os.path.join(project_root, "config.yaml")
    default_stage1_root = os.path.join(os.path.dirname(__file__), "stage1_swap_results")
    default_stage2_root = os.path.join(os.path.dirname(__file__), "stage2_swap_results")
    default_save_root = os.path.join(os.path.dirname(__file__), "swap_vis")

    parser = argparse.ArgumentParser(description="Render swap experiment visualizations.")
    parser.add_argument("--config", type=str, default=default_config, help="Path to config.yaml")
    parser.add_argument("--stage", type=str, default="stage2", choices=["stage1", "stage2", "all"])
    parser.add_argument(
        "--pair_type",
        type=str,
        default="text_emotion",
        choices=["text_emotion", "text_intensity", "identity", "all"],
    )
    parser.add_argument("--group_name", type=str, default=None, help="Render a specific group_xxxxx")
    parser.add_argument("--max_groups", type=int, default=-1, help="-1 means all groups")
    parser.add_argument("--gpu", type=int, default=None)

    parser.add_argument("--stage1_root", type=str, default=default_stage1_root)
    parser.add_argument("--stage2_root", type=str, default=default_stage2_root)
    parser.add_argument("--save_root", type=str, default=default_save_root)

    parser.add_argument("--num_frames", type=int, default=4, help="How many frames to sample uniformly")

    # 内部高分辨率渲染
    parser.add_argument("--render_width", type=int, default=1400, help="Internal render width")
    parser.add_argument("--render_height", type=int, default=1000, help="Internal render height")

    # 最终输出格子尺寸
    parser.add_argument("--cell_width", type=int, default=220, help="Output cell width")
    parser.add_argument("--cell_height", type=int, default=360, help="Output cell height")

    # 更紧凑的布局
    parser.add_argument("--row_gap", type=int, default=2, help="Vertical gap between rows")
    parser.add_argument("--col_gap", type=int, default=2, help="Horizontal gap between columns")
    parser.add_argument("--title_height", type=int, default=34, help="Top title area height")
    parser.add_argument("--meta_height", type=int, default=34, help="Source/target meta area height")
    parser.add_argument("--col_header_height", type=int, default=26, help="Column header area height")

    parser.add_argument("--crop_padding", type=int, default=6, help="Padding after tight crop")
    parser.add_argument("--white_threshold", type=int, default=245, help="Background threshold for auto crop")

    parser.add_argument(
        "--font_path",
        type=str,
        default=None,
        help="Optional font path. If None, try Times New Roman then serif fallbacks."
    )

    parser.add_argument(
        "--flame_model_path",
        type=str,
        default="/home/chensheng/1Project/Project2/FLAME/flame_model/generic_model.pkl",
        help="FLAME model pkl path"
    )
    parser.add_argument(
        "--static_landmark_embedding_path",
        type=str,
        default="/home/chensheng/1Project/Project2/FLAME/flame_model/flame_static_embedding.pkl",
        help="FLAME static landmark embedding path"
    )
    parser.add_argument(
        "--dynamic_landmark_embedding_path",
        type=str,
        default="/home/chensheng/1Project/Project2/FLAME/flame_model/flame_dynamic_embedding.npy",
        help="FLAME dynamic landmark embedding path"
    )
    parser.add_argument(
        "--template_path",
        type=str,
        default="/home/chensheng/1Project/Project2/FLAME/flame_sample.ply",
        help="Template mesh path"
    )

    parser.add_argument("--strict", action="store_true", help="Raise exception immediately instead of skipping")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing npy file: {path}")
    return np.load(path)


def build_dataset_and_index(test_dir: str):
    dataset = CustomDataset(test_dir)
    token_to_index = {}
    for idx, file_path in enumerate(dataset.files):
        token = dataset.extract_video_token(file_path)
        if token not in token_to_index:
            token_to_index[token] = idx
    return dataset, token_to_index


def ensure_tensor(x, device: torch.device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)


def load_shape_from_token(
    dataset: CustomDataset,
    token_to_index: Dict[str, int],
    video_token: str,
    device: torch.device,
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

        cfg = SimpleNamespace(
            flame_model_path=self.flame_model_path,
            static_landmark_embedding_path=self.static_landmark_embedding_path,
            dynamic_landmark_embedding_path=self.dynamic_landmark_embedding_path,
            batch_size=1,
            use_face_contour=False,
            shape_params=int(shape_dim),
            expression_params=int(exp_dim),
            use_3D_translation=False,
        )
        flame = FLAME(cfg).to(self.device)
        flame.eval()
        self.cache[key] = flame
        return flame


def decode_vertices_sequence(
    shape_arr: np.ndarray,
    exp_arr: np.ndarray,
    jaw_arr: np.ndarray,
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


class MeshRenderer:
    def __init__(self, template_path: str, width: int = 1400, height: int = 1000):
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template mesh not found: {template_path}")

        self.template_mesh = trimesh.load_mesh(template_path)
        self.faces = self.template_mesh.faces.copy()

        self.cam = pyrender.PerspectiveCamera(yfov=np.pi / 20, aspectRatio=1.414)
        self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)

        self.camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.6],
            [0.0, 0.0, 0.0, 1.0],
        ])

        self.renderer = pyrender.OffscreenRenderer(width, height)

    def render_frame(self, vertices: np.ndarray) -> np.ndarray:
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices should be [V,3], got {vertices.shape}")

        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        py_mesh = pyrender.Mesh.from_trimesh(mesh)

        scene = pyrender.Scene(bg_color=[255, 255, 255], ambient_light=[0.3, 0.3, 0.3])
        scene.add(py_mesh)
        scene.add(self.cam, pose=self.camera_pose)
        scene.add(self.light, pose=self.camera_pose)

        color, _ = self.renderer.render(scene)
        return color

    def close(self):
        self.renderer.delete()


def get_font(font_size: int, font_path: str = None):
    candidate_paths = []

    if font_path is not None:
        candidate_paths.append(font_path)

    candidate_paths.extend([
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesnewroman.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ])

    for fp in candidate_paths:
        if fp is not None and os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, font_size)
            except Exception:
                pass

    return ImageFont.load_default()


def draw_centered_text(
    canvas: np.ndarray,
    text: str,
    y0: int,
    height: int,
    x0: int = 0,
    width: int = None,
    font_size: int = 24,
    color=(0, 0, 0),
    font_path: str = None,
):
    if width is None:
        width = canvas.shape[1]

    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    font = get_font(font_size, font_path)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = x0 + max((width - text_w) // 2, 0)
    y = y0 + max((height - text_h) // 2, 0)

    draw.text((x, y), text, fill=color, font=font)
    return np.array(pil_img)


def draw_left_text(
    canvas: np.ndarray,
    text: str,
    y0: int,
    height: int,
    x_pad: int = 8,
    font_size: int = 15,
    color=(0, 0, 0),
    font_path: str = None,
):
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    font = get_font(font_size, font_path)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_h = bbox[3] - bbox[1]
    y = y0 + max((height - text_h) // 2, 0)

    draw.text((x_pad, y), text, fill=color, font=font)
    return np.array(pil_img)


def tight_crop_face(
    img: np.ndarray,
    padding: int = 6,
    white_threshold: int = 245,
) -> np.ndarray:
    if img.ndim != 3:
        return img

    fg_mask = np.any(img < white_threshold, axis=2)
    ys, xs = np.where(fg_mask)

    if len(xs) == 0 or len(ys) == 0:
        return img

    y_min = max(int(ys.min()) - padding, 0)
    y_max = min(int(ys.max()) + padding + 1, img.shape[0])
    x_min = max(int(xs.min()) - padding, 0)
    x_max = min(int(xs.max()) + padding + 1, img.shape[1])

    return img[y_min:y_max, x_min:x_max]


def resize_with_aspect_and_pad(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    bg_color: int = 255
) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def postprocess_rendered_frame(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    crop_padding: int,
    white_threshold: int,
) -> np.ndarray:
    img = tight_crop_face(
        img,
        padding=crop_padding,
        white_threshold=white_threshold,
    )
    img = resize_with_aspect_and_pad(
        img,
        target_w=target_w,
        target_h=target_h,
        bg_color=255,
    )
    return img


def get_frame_indices(seq_len: int, num_frames: int) -> List[int]:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if num_frames <= 1:
        return [seq_len // 2]

    idxs = np.linspace(0, seq_len - 1, num=num_frames)
    idxs = np.round(idxs).astype(int).tolist()
    return sorted(list(dict.fromkeys(idxs)))


def make_grid(rows: List[List[np.ndarray]], row_gap: int = 2, col_gap: int = 2, bg_color: int = 255) -> np.ndarray:
    num_rows = len(rows)
    num_cols = len(rows[0])

    h = rows[0][0].shape[0]
    w = rows[0][0].shape[1]
    c = rows[0][0].shape[2]

    canvas_h = num_rows * h + (num_rows + 1) * row_gap
    canvas_w = num_cols * w + (num_cols + 1) * col_gap
    canvas = np.full((canvas_h, canvas_w, c), bg_color, dtype=np.uint8)

    for r in range(num_rows):
        for cidx in range(num_cols):
            y0 = row_gap + r * (h + row_gap)
            x0 = col_gap + cidx * (w + col_gap)
            canvas[y0:y0 + h, x0:x0 + w] = rows[r][cidx]

    return canvas


def get_result_paths(stage: str, group_dir: str):
    if stage == "stage1":
        return {
            "source_orig_exp": os.path.join(group_dir, "source_rec_exp.npy"),
            "source_orig_jaw": os.path.join(group_dir, "source_rec_jaw.npy"),
            "target_orig_exp": os.path.join(group_dir, "target_rec_exp.npy"),
            "target_orig_jaw": os.path.join(group_dir, "target_rec_jaw.npy"),
        }
    if stage == "stage2":
        return {
            "source_orig_exp": os.path.join(group_dir, "source_gen_exp.npy"),
            "source_orig_jaw": os.path.join(group_dir, "source_gen_jaw.npy"),
            "target_orig_exp": os.path.join(group_dir, "target_gen_exp.npy"),
            "target_orig_jaw": os.path.join(group_dir, "target_gen_jaw.npy"),
        }
    raise ValueError(f"Unsupported stage: {stage}")


def render_one_group(
    stage: str,
    pair_type: str,
    group_dir: str,
    dataset: CustomDataset,
    token_to_index: Dict[str, int],
    flame_cache: FlameDecoderCache,
    mesh_renderer: MeshRenderer,
    device: torch.device,
    save_dir: str,
    num_frames: int,
    args,
):
    meta_path = os.path.join(group_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")

    meta = load_json(meta_path)
    result_paths = get_result_paths(stage, group_dir)

    source_token = meta["source_video_token"]
    target_token = meta["target_video_token"]

    source_shape = load_shape_from_token(dataset, token_to_index, source_token, device)
    target_shape = load_shape_from_token(dataset, token_to_index, target_token, device)

    source_orig_exp = load_npy(result_paths["source_orig_exp"])
    source_orig_jaw = load_npy(result_paths["source_orig_jaw"])
    target_orig_exp = load_npy(result_paths["target_orig_exp"])
    target_orig_jaw = load_npy(result_paths["target_orig_jaw"])

    source_swap_exp = load_npy(os.path.join(group_dir, "source_swap_exp.npy"))
    source_swap_jaw = load_npy(os.path.join(group_dir, "source_swap_jaw.npy"))
    target_swap_exp = load_npy(os.path.join(group_dir, "target_swap_exp.npy"))
    target_swap_jaw = load_npy(os.path.join(group_dir, "target_swap_jaw.npy"))

    source_orig_vertices = decode_vertices_sequence(source_shape, source_orig_exp, source_orig_jaw, flame_cache, device)
    target_orig_vertices = decode_vertices_sequence(target_shape, target_orig_exp, target_orig_jaw, flame_cache, device)
    source_swap_vertices = decode_vertices_sequence(source_shape, source_swap_exp, source_swap_jaw, flame_cache, device)
    target_swap_vertices = decode_vertices_sequence(target_shape, target_swap_exp, target_swap_jaw, flame_cache, device)

    T = min(
        len(source_orig_vertices),
        len(target_orig_vertices),
        len(source_swap_vertices),
        len(target_swap_vertices),
    )
    frame_indices = get_frame_indices(T, num_frames)

    rows = []
    for frame_idx in frame_indices:
        img_source_orig = mesh_renderer.render_frame(source_orig_vertices[frame_idx])
        img_target_orig = mesh_renderer.render_frame(target_orig_vertices[frame_idx])
        img_source_swap = mesh_renderer.render_frame(source_swap_vertices[frame_idx])
        img_target_swap = mesh_renderer.render_frame(target_swap_vertices[frame_idx])

        img_source_orig = postprocess_rendered_frame(
            img_source_orig,
            target_w=args.cell_width,
            target_h=args.cell_height,
            crop_padding=args.crop_padding,
            white_threshold=args.white_threshold,
        )
        img_target_orig = postprocess_rendered_frame(
            img_target_orig,
            target_w=args.cell_width,
            target_h=args.cell_height,
            crop_padding=args.crop_padding,
            white_threshold=args.white_threshold,
        )
        img_source_swap = postprocess_rendered_frame(
            img_source_swap,
            target_w=args.cell_width,
            target_h=args.cell_height,
            crop_padding=args.crop_padding,
            white_threshold=args.white_threshold,
        )
        img_target_swap = postprocess_rendered_frame(
            img_target_swap,
            target_w=args.cell_width,
            target_h=args.cell_height,
            crop_padding=args.crop_padding,
            white_threshold=args.white_threshold,
        )

        rows.append([img_source_orig, img_target_orig, img_source_swap, img_target_swap])

    grid = make_grid(
        rows,
        row_gap=args.row_gap,
        col_gap=args.col_gap,
        bg_color=255
    )

    total_header_h = args.title_height + args.meta_height + args.col_header_height
    header = np.full((total_header_h, grid.shape[1], 3), 255, dtype=np.uint8)

    title = f"{stage} | {pair_type} | {Path(group_dir).name}"
    header = draw_centered_text(
        header,
        text=title,
        y0=0,
        height=args.title_height,
        font_size=22,
        color=(0, 0, 0),
        font_path=args.font_path,
    )

    src_line = f"Source: {source_token}"
    tgt_line = f"Target: {target_token}"
    meta_y0 = args.title_height
    meta_h = args.meta_height

    header = draw_left_text(
        header,
        text=src_line,
        y0=meta_y0,
        height=meta_h // 2,
        x_pad=8,
        font_size=15,
        color=(0, 0, 0),
        font_path=args.font_path,
    )
    header = draw_left_text(
        header,
        text=tgt_line,
        y0=meta_y0 + meta_h // 2,
        height=meta_h - meta_h // 2,
        x_pad=8,
        font_size=15,
        color=(0, 0, 0),
        font_path=args.font_path,
    )

    col_header_y0 = meta_y0 + meta_h
    cell_w = rows[0][0].shape[1]
    col_titles = ["Source Original", "Target Original", "Source Swapped", "Target Swapped"]

    for i, col_title in enumerate(col_titles):
        x0 = args.col_gap + i * (cell_w + args.col_gap)
        header = draw_centered_text(
            header,
            text=col_title,
            y0=col_header_y0,
            height=args.col_header_height,
            x0=x0,
            width=cell_w,
            font_size=16,
            color=(0, 0, 0),
            font_path=args.font_path,
        )

    vis = np.concatenate([header, grid], axis=0)

    ensure_dir(save_dir)
    out_path = os.path.join(save_dir, f"{Path(group_dir).name}.png")
    ok = cv2.imwrite(out_path, vis)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {out_path}")

    print(f"[Saved] {out_path}")


def list_group_dirs(root_dir: str) -> List[str]:
    if not os.path.exists(root_dir):
        return []
    dirs = []
    for name in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and name.startswith("group_"):
            dirs.append(full)
    return dirs


def select_group_dirs(result_root: str, group_name: str = None, max_groups: int = -1) -> List[str]:
    if group_name is not None:
        return [os.path.join(result_root, group_name)]

    all_group_dirs = list_group_dirs(result_root)
    if max_groups is None or max_groups < 0:
        return all_group_dirs
    return all_group_dirs[:max_groups]


def main():
    args = parse_args()
    config = load_yaml(args.config)

    device = torch.device(
        f"cuda:{args.gpu if args.gpu is not None else config['predict']['gpu']}"
        if torch.cuda.is_available() else "cpu"
    )

    print("=" * 80)
    print(f"device                           : {device}")
    print(f"stage                            : {args.stage}")
    print(f"pair_type                        : {args.pair_type}")
    print(f"flame_model_path                 : {args.flame_model_path}")
    print(f"static_landmark_embedding_path   : {args.static_landmark_embedding_path}")
    print(f"dynamic_landmark_embedding_path  : {args.dynamic_landmark_embedding_path}")
    print(f"template_path                    : {args.template_path}")
    print("=" * 80)

    test_dir = config["test_file_path"]
    dataset, token_to_index = build_dataset_and_index(test_dir)

    if args.stage == "all":
        stages = ["stage1", "stage2"]
    else:
        stages = [args.stage]

    if args.pair_type == "all":
        pair_types = ["text_emotion", "text_intensity", "identity"]
    else:
        pair_types = [args.pair_type]

    flame_cache = FlameDecoderCache(
        flame_model_path=args.flame_model_path,
        static_landmark_embedding_path=args.static_landmark_embedding_path,
        dynamic_landmark_embedding_path=args.dynamic_landmark_embedding_path,
        device=device,
    )

    mesh_renderer = MeshRenderer(
        template_path=args.template_path,
        width=args.render_width,
        height=args.render_height,
    )

    try:
        for stage in stages:
            stage_root = args.stage1_root if stage == "stage1" else args.stage2_root

            for pair_type in pair_types:
                result_root = os.path.join(stage_root, pair_type)
                group_dirs = select_group_dirs(
                    result_root=result_root,
                    group_name=args.group_name,
                    max_groups=args.max_groups,
                )

                print("=" * 80)
                print(f"result_root                      : {result_root}")
                print(f"group_dirs                       : {group_dirs[:5]}{' ...' if len(group_dirs) > 5 else ''}")
                print(f"num_group_dirs                   : {len(group_dirs)}")
                print("=" * 80)

                save_dir = os.path.join(args.save_root, stage, pair_type)
                ensure_dir(save_dir)

                for group_dir in group_dirs:
                    print(f"[Start] {group_dir}")
                    if not os.path.exists(group_dir):
                        msg = f"group dir does not exist: {group_dir}"
                        if args.strict:
                            raise FileNotFoundError(msg)
                        print(f"[Skip] {msg}")
                        continue

                    if args.strict:
                        render_one_group(
                            stage=stage,
                            pair_type=pair_type,
                            group_dir=group_dir,
                            dataset=dataset,
                            token_to_index=token_to_index,
                            flame_cache=flame_cache,
                            mesh_renderer=mesh_renderer,
                            device=device,
                            save_dir=save_dir,
                            num_frames=args.num_frames,
                            args=args,
                        )
                    else:
                        try:
                            render_one_group(
                                stage=stage,
                                pair_type=pair_type,
                                group_dir=group_dir,
                                dataset=dataset,
                                token_to_index=token_to_index,
                                flame_cache=flame_cache,
                                mesh_renderer=mesh_renderer,
                                device=device,
                                save_dir=save_dir,
                                num_frames=args.num_frames,
                                args=args,
                            )
                        except Exception as e:
                            print(f"[Skip] {group_dir} because of error: {e}")
                            traceback.print_exc()
    finally:
        mesh_renderer.close()

    print("=" * 80)
    print("Render swap visualization finished.")
    print(f"Saved to: {args.save_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()