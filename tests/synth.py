"""合成測試素材：由真實照片產生「已知真值」的多段曝光＋燈光組。
相機曲線刻意不是 sRGB，加入雜訊、8-bit 量化與次像素位移。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
from retouch.imageio import load_image, srgb_to_linear, linear_to_srgb, save_jpg

ROOT = Path(__file__).resolve().parent.parent


def camera_curve(x: np.ndarray) -> np.ndarray:
    v = linear_to_srgb(np.clip(x, 0, 1))
    return np.clip(v + 0.06 * np.sin(2 * np.pi * v) / (2 * np.pi) * -1 + 0.0, 0, 1)


def scene(name: str, width: int = 1024) -> np.ndarray:
    R0 = srgb_to_linear(load_image(ROOT / name, width))
    t = np.clip((R0.mean(-1, keepdims=True) - 0.55) / 0.4, 0, 1)
    return (R0 * (1 + 14 * t * t)).astype(np.float32)  # 亮處（窗外）拉成 HDR


def blob(shape, cx, cy, sx, sy, amp):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((xx - cx * w) / (sx * w)) ** 2 + ((yy - cy * h) / (sy * h)) ** 2) / 2)
    return (amp * g)[..., None] * np.array([1.0, 0.85, 0.6], np.float32)  # 暖色燈


def shoot(R, ev, dx, dy, rng, ref_scale):
    x = R * ref_scale * 2.0 ** ev
    x = x + rng.normal(0, 0.002, x.shape).astype(np.float32) * np.sqrt(np.clip(x, 0, 1) + 0.01)
    v = camera_curve(x)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    v = cv2.warpAffine(v, M, (v.shape[1], v.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.round(v * 255) / 255


def make_dataset(out: Path, seed: int = 0) -> dict:
    """回傳真值：各組檔名、ev、位移、光狀態、燈光層。"""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    truth = {"groups": []}
    # 吧檯組：環境光 5 段 + 吊燈 3 + 軌道燈 2 + 層板燈 2（暗張多）
    R = scene("5.jpg")
    alb = np.clip(srgb_to_linear(load_image(ROOT / "5.jpg", 1024)), 0.05, 1)
    lamps = {
        "A": blob(R.shape, 0.55, 0.35, 0.16, 0.20, 0.5) * alb,
        "B": blob(R.shape, 0.30, 0.45, 0.025, 0.04, 3.0) * alb,
        "C": blob(R.shape, 0.62, 0.72, 0.10, 0.03, 1.2) * alb,
    }
    shots = [("amb", e) for e in (2, 1, 0, -1, -2)] + [("A", -1), ("A", -2.5), ("A", -1)] \
        + [("B", -2.5), ("B", -3)] + [("C", -1), ("C", -3)]
    files, evs, shifts, st = [], [], [], []
    for k, (s, e) in enumerate(shots):
        dx, dy = (0, 0) if k == 0 else rng.uniform(-4, 4, 2)
        Rs = R + lamps[s] if s != "amb" else R
        fn = f"bar_{2712 + k}.jpg"
        save_jpg(out / fn, shoot(Rs, e, dx, dy, rng, 0.25))
        files.append(fn); evs.append(e); shifts.append((float(dx), float(dy))); st.append(s)
    truth["groups"].append(dict(files=files, ev=evs, shift=shifts, state=st, R=R, lamps=lamps))
    # 客廳組：4 段曝光，無燈光變化
    R2 = scene("7.jpg")
    files, evs, shifts = [], [], []
    for k, e in enumerate((1, 0, -1, -2.5)):
        dx, dy = (0, 0) if k == 0 else rng.uniform(-3, 3, 2)
        fn = f"living_{2780 + k}.jpg"
        save_jpg(out / fn, shoot(R2, e, dx, dy, rng, 0.25))
        files.append(fn); evs.append(e); shifts.append((float(dx), float(dy)))
    truth["groups"].append(dict(files=files, ev=evs, shift=shifts, state=["amb"] * 4, R=R2))
    # 單張
    save_jpg(out / "single_1.jpg", load_image(ROOT / "1.jpg", 1024))
    truth["groups"].append(dict(files=["single_1.jpg"]))
    return truth
