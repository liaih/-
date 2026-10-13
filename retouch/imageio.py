"""讀寫影像。內部一律為 float32 RGB [0,1]。
JPG/TIFF 為「已編碼」（有相機曲線）；CR3 經 rawpy 為線性，載入後以 sRGB 編碼，
讓後續校準流程一致（校準曲線會自動退化為近似 sRGB 解碼）。
Windows 中文路徑：用 np.fromfile / tofile，不直接用 cv2.imread。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2

RAW_EXT = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
IMG_EXT = {".jpg", ".jpeg", ".tif", ".tiff", ".png"} | RAW_EXT


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0, None)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def luma(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def _read_raw(path: Path) -> np.ndarray:
    import rawpy
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                              gamma=(1, 1), output_color=rawpy.ColorSpace.sRGB)
    return linear_to_srgb(rgb.astype(np.float32) / 65535.0)


def load_image(path: str | Path, max_side: int | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in RAW_EXT:
        rgb = _read_raw(path)
    else:
        buf = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if bgr is None:
            raise IOError(f"無法讀取：{path}")
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        bgr = bgr[..., :3]
        scale = 65535.0 if bgr.dtype == np.uint16 else 255.0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / scale
    if max_side and max(rgb.shape[:2]) > max_side:
        rgb = resize_max(rgb, max_side)
    return rgb


def resize_max(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return rgb
    return cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def save_jpg(path: str | Path, rgb: np.ndarray, quality: int = 95) -> None:
    bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError(f"JPG 編碼失敗：{path}")
    buf.tofile(str(path))


def save_tiff16(path: str | Path, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".tif", bgr)
    if not ok:
        raise IOError(f"TIFF 編碼失敗：{path}")
    buf.tofile(str(path))
