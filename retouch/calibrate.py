"""風格常數校準：以使用者的 LR 成品為準，擬合 style.K 的常數（不改 xmp 數值）。
流程：同機位原圖 → HDR 合成（縮圖）→ 以 SIFT homography 把成品對到我們的畫面
（成品有 Upright 裁切）→ Nelder–Mead 最小化 Lab 差異 → 存 presets/style_k.json。"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import cv2
from .imageio import list_images, load_image, resize_max, luma
from .grouping import group_by_angle
from .pipeline import analyze_group, PRESETS
from .merge import hdr_merge
from .style import K, apply_style, auto_exposure, exposure_metric
from .xmp import parse_xmp

# 要擬合的常數與合理範圍
FIT = {"temp": (0.0, 0.05), "tint": (0.0, 0.05), "shadows": (0.2, 3.0), "highlights": (0.2, 3.0),
       "blacks": (0.0, 2.0), "whites": (0.0, 1.5), "contrast": (0.0, 1.5),
       "vibrance": (0.0, 3.0), "saturation": (0.0, 3.0), "hsl_sat": (0.0, 2.0),
       "hdr_knee": (0.2, 2.0), "hdr_slope": (0.1, 1.0)}
STYLE_K_FILE = PRESETS / "style_k.json"


def _gray8(x):
    return (np.clip(luma(x), 0, 1) * 255).astype(np.uint8)


def register(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """把 ref 對到 src 的座標。回傳 (對齊後 ref, 有效遮罩, inlier 數)。"""
    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(_gray8(src), None)
    k2, d2 = sift.detectAndCompute(_gray8(ref), None)
    good = [a for a, b in cv2.BFMatcher().knnMatch(d2, d1, k=2) if a.distance < 0.7 * b.distance]
    if len(good) < 12:
        raise RuntimeError("成品與原圖特徵點對不上，請確認是同一機位")
    H, inl = cv2.findHomography(np.float32([k2[a.queryIdx].pt for a in good]),
                                np.float32([k1[a.trainIdx].pt for a in good]), cv2.RANSAC, 3)
    h, w = src.shape[:2]
    out = cv2.warpPerspective(ref, H, (w, h))
    m = cv2.warpPerspective(np.ones(ref.shape[:2], np.float32), H, (w, h)) > 0.99
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return out, m, int(inl.sum())


def _lab(x):
    return cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def _nelder_mead(f, x0, step, iters=300, tol=1e-5):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], 0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def calibrate_style(bracket_dir, finished_jpg, style_path=PRESETS / "style_default.xmp",
                    width: int = 400, log=print) -> dict:
    p = parse_xmp(style_path)
    paths = list_images(bracket_dir)
    thumbs = [load_image(q, 1024) for q in paths]
    groups, _ = group_by_angle(thumbs)
    idx = max(groups, key=len)
    A = analyze_group([thumbs[i] for i in idx])
    st = A["states"]
    R, _ = hdr_merge(((i, A["vs"][i]) for i in st.members[st.ambient]), A["cal"].curves, A["cal"].ev)
    ref = load_image(finished_jpg, 1024)
    target = exposure_metric(ref)
    k0 = dict(K)
    ev0 = auto_exposure(resize_max(R, 768), p, target, k0)
    ref_w, m, n_in = register(apply_style(R, p, ev0, k0), ref)
    log(f"對位 inliers={n_in}，重疊 {m.mean():.0%}")
    s = width / R.shape[1]
    small = lambda x: cv2.resize(x, (width, round(R.shape[0] * s)), interpolation=cv2.INTER_AREA)
    Rs, ref_lab = small(R), _lab(small(ref_w))
    ms = small(m.astype(np.float32)) > 0.99
    names = list(FIT)
    lo = np.array([FIT[n][0] for n in names]); hi = np.array([FIT[n][1] for n in names])

    def unpack(x):
        k = dict(k0); k.update({n: float(v) for n, v in zip(names, np.clip(x[:-1], lo, hi))})
        return k, float(x[-1])

    def loss(x):
        k, ev = unpack(x)
        d = _lab(apply_style(Rs, p, ev, k))[ms] - ref_lab[ms]
        return float(np.mean(np.abs(d) * np.array([1.0, 1.5, 1.5])))  # 色偏權重較高

    x0 = np.array([k0[n] for n in names] + [ev0])
    l0 = loss(x0)
    x, l1 = _nelder_mead(loss, x0, np.r_[(hi - lo) * 0.15, 0.3], iters=600)
    k1, _ = unpack(x)
    log(f"Lab 平均誤差 {l0:.2f} → {l1:.2f}")
    fitted = {n: round(k1[n], 4) for n in names}
    STYLE_K_FILE.write_text(json.dumps({"K": fitted, "finished": Path(finished_jpg).name,
                                        "loss_before": round(l0, 3), "loss_after": round(l1, 3)},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
    return dict(K=fitted, loss_before=l0, loss_after=l1, target=target)
