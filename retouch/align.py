"""對齊：以曝光不變的結構圖做 ECC，估計平移（腳架拍攝只需微調）。"""
from __future__ import annotations
import numpy as np
import cv2
from .grouping import structure_map


def estimate_shift(ref: np.ndarray, img: np.ndarray, width: int = 640) -> tuple[np.ndarray, float]:
    """回傳 (2x3 warp 以全尺寸像素為單位, 相關係數)。使用 translation 模型。"""
    a = structure_map(ref, width)
    b = structure_map(img, width)
    scale = ref.shape[1] / a.shape[1]
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    try:
        cc, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_TRANSLATION, crit, None, 5)
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), 0.0
    warp = warp.copy()
    warp[:, 2] *= scale
    return warp, float(cc)


def apply_warp(img: np.ndarray, warp: np.ndarray) -> np.ndarray:
    # 位移小於半像素時不重取樣：次像素內插會讓每張都變糊，疊加後更明顯，代價大於對位收益
    if abs(warp[0, 2]) < 0.5 and abs(warp[1, 2]) < 0.5:
        return img
    h, w = img.shape[:2]
    return cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def chain_align(images: list[np.ndarray], min_cc: float = 0.6) -> tuple[list[np.ndarray], list[dict]]:
    """逐張對齊：每張對「最相似且已對齊」的鄰居，位移累加到基準張。
    暗圖、燈光不同的圖不直接對基準，避免被亮區差異帶偏。"""
    from .grouping import structure_map, similarity
    n = len(images)
    maps = [structure_map(im, 360) for im in images]
    S = np.array([[similarity(maps[i], maps[j]) for j in range(n)] for i in range(n)])
    ref = int(np.argmax(S.sum(1)))
    shifts = {ref: np.zeros(2)}
    info = {ref: dict(cc=1.0, to=ref)}
    pending = set(range(n)) - {ref}
    while pending:
        i, j = max(((i, j) for i in pending for j in shifts), key=lambda t: S[t])
        w, cc = estimate_shift(images[j], images[i])
        d = w[:, 2] if cc >= min_cc else np.zeros(2)
        shifts[i] = shifts[j] + d
        info[i] = dict(cc=cc, to=j, trusted=cc >= min_cc)
        pending.remove(i)
    warps, report = [], []
    for i in range(n):
        w = np.eye(2, 3, dtype=np.float32); w[:, 2] = shifts[i]
        warps.append(w)
        report.append(dict(index=i, dx=float(shifts[i][0]), dy=float(shifts[i][1]), **info[i]))
    return warps, report
