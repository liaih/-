"""自動分群：不依賴 EXIF，只看畫面結構。同一機位 = 結構相似；曝光與燈光不同不影響。"""
from __future__ import annotations
import numpy as np
import cv2
from .imageio import luma


def structure_map(rgb: np.ndarray, width: int = 360) -> np.ndarray:
    """曝光不變的結構圖：log 亮度 → 去除低頻 → 局部對比正規化。"""
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    y = luma(small)
    y = np.log(y + 0.01 * (y.max() + 1e-3) + 1e-4)
    low = cv2.GaussianBlur(y, (0, 0), 6)
    d = y - low
    sd = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), 6)) + 1e-3 * (1 + np.abs(d).mean())
    return (d / sd).astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def group_by_angle(images: list[np.ndarray], threshold: float = 0.35) -> list[list[int]]:
    maps = [structure_map(im) for im in images]
    n = len(images)
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = similarity(maps[i], maps[j])
    # 單連結聚類
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in sorted(groups.values(), key=lambda g: g[0])], S
