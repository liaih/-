"""一鍵風格：以 xmp 數值近似 Lightroom 的基本調整（非 Adobe 演算法，為近似值）。
輸入為線性輻射量（HDR，可 >1），輸出 sRGB 編碼 [0,1]。
順序：白平衡增量 → 曝光（xmp + 自動）→ 亮部／陰影（大半徑平滑底層的局部增益）
      → sRGB 編碼 → 白色／黑色／對比（亮度曲線，保色相）→ 自然飽和度／飽和度／HSL。
常數集中在 K，待與 LR 成品（2-4.jpg）對照微調。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
from .imageio import luma, linear_to_srgb, load_image

K = dict(
    temp=0.004,        # 每單位色溫增量的 log2 增益（R/B 反向）
    tint=0.003,        # 每單位色調增量的 log2 增益（G）
    shadows=1.3,       # 陰影 +100 時暗部提亮 EV
    highlights=1.5,    # 亮部 -100 時亮部壓暗 EV
    base_radius=0.04,  # 底層平滑半徑（佔短邊比例）
    whites=0.35, blacks=0.9, contrast=0.6,
    vibrance=1.0, saturation=1.0, hsl_sat=1.0, hsl_lum=0.35, hsl_hue=30.0,
)
HSL_CENTERS = [0, 30, 60, 120, 180, 240, 270, 300]
HSL_COLORS = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _guided_base(y: np.ndarray, radius_frac: float, eps: float = 0.02) -> np.ndarray:
    """快速 guided filter（自身為導引），在縮圖上算係數再放大。"""
    h, w = y.shape
    s = min(1.0, 512 / max(h, w))
    ys = cv2.resize(y, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else y
    r = max(2, round(radius_frac * min(ys.shape)))
    box = lambda a: cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
    m = box(ys); var = box(ys * ys) - m * m
    a = var / (var + eps); b = m - a * m
    a, b = box(a), box(b)
    if s < 1:
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return a * y + b


def _hue_weights(H: np.ndarray) -> np.ndarray:
    """8 個 LR 色相中心的三角形權重（環狀），回傳 (..., 8)。"""
    c = np.array(HSL_CENTERS + [360], float)
    W = np.zeros(H.shape + (8,), np.float32)
    for k in range(8):
        lo, hi = c[k], c[k + 1]
        t = ((H - lo) / (hi - lo)).astype(np.float32)
        inside = (H >= lo) & (H < hi)
        W[..., k] += np.where(inside, 1 - t, 0)
        W[..., (k + 1) % 8] += np.where(inside, t, 0)
    return W


def wb_gains(p: dict, k: dict = K) -> np.ndarray:
    t, n = p.get("IncrementalTemperature", 0), p.get("IncrementalTint", 0)
    g = np.array([2 ** (k["temp"] * t), 2 ** (-k["tint"] * n), 2 ** (-k["temp"] * t)], np.float32)
    return g / (g @ np.array([0.2126, 0.7152, 0.0722], np.float32))


def apply_style(R: np.ndarray, p: dict, ev: float = 0.0, k: dict = K) -> np.ndarray:
    x = R.astype(np.float32) * wb_gains(p, k) * np.float32(2 ** (p.get("Exposure2012", 0) + ev))
    # 亮部／陰影：依平滑底層決定局部 EV 增益，保留細節
    yb = np.clip(linear_to_srgb(luma(x)), 0, 1.5)
    base = _guided_base(yb, k["base_radius"])
    ws = 1 - _smoothstep(0.05, 0.6, base)
    wh = _smoothstep(0.45, 1.05, base)
    loc = p.get("Shadows2012", 0) / 100 * k["shadows"] * ws + p.get("Highlights2012", 0) / 100 * k["highlights"] * wh
    x *= np.exp2(loc)[..., None]
    g = linear_to_srgb(x)
    # 亮度曲線：白色、黑色、對比（以亮度比例套用，維持色相）
    Y = luma(g)
    Yc = np.clip(Y, 0, None)
    w, b, c = p.get("Whites2012", 0) / 100, p.get("Blacks2012", 0) / 100, p.get("Contrast2012", 0) / 100
    T = Yc * (1 + k["whites"] * w * np.clip(Yc, 0, 1.2) ** 3)
    T = T * (1 + k["blacks"] * b * (1 - np.clip(T, 0, 1)) ** 4)
    t = np.clip(T, 0, 1)
    T = np.where(T <= 1, t - k["contrast"] * c * np.sin(2 * np.pi * t) / (2 * np.pi), T)
    g = g * (T / np.maximum(Y, 1e-5))[..., None]
    g = np.clip(g, 0, 1).astype(np.float32)
    # 色彩：HSV 空間
    hsv = cv2.cvtColor(g, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    Wh = _hue_weights(H)
    hs = np.array([p.get(f"SaturationAdjustment{n}", 0) for n in HSL_COLORS], np.float32) / 100
    hh = np.array([p.get(f"HueAdjustment{n}", 0) for n in HSL_COLORS], np.float32) / 100
    hl = np.array([p.get(f"LuminanceAdjustment{n}", 0) for n in HSL_COLORS], np.float32) / 100
    f = (1 + k["saturation"] * p.get("Saturation", 0) / 100) \
        * (1 + k["vibrance"] * p.get("Vibrance", 0) / 100 * (1 - S)) \
        * (1 + k["hsl_sat"] * (Wh @ hs))
    if hh.any():
        H = (H + k["hsl_hue"] * (Wh @ hh)) % 360
    if hl.any():
        V = V * (1 + k["hsl_lum"] * (Wh @ hl) * S)
    hsv = np.stack([H, np.clip(S * f, 0, 1), np.clip(V, 0, 1)], -1).astype(np.float32)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1)


# ---------- 自動曝光 ----------
DEFAULT_TARGET = 0.62  # 套風格後的亮度中位數（sRGB 編碼）；建議用 learn-target 由成品學習


def learn_target(paths: list[str | Path]) -> float:
    """由使用者的成品 JPG 學習目標亮度中位數。"""
    return float(np.median([np.median(luma(load_image(p, 1024))) for p in paths]))


def auto_exposure(R: np.ndarray, p: dict, target: float = DEFAULT_TARGET, k: dict = K,
                  lo: float = -6.0, hi: float = 6.0, iters: int = 22) -> float:
    """二分搜尋 ev，使 apply_style 後的亮度中位數 = target。R 建議先縮到 ~768px。"""
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = float(np.median(luma(apply_style(R, p, mid, k))))
        lo, hi = (mid, hi) if m < target else (lo, mid)
    return (lo + hi) / 2
