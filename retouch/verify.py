"""自動化驗收：逐案件跑完整流程，輸出「原圖｜APP 成品｜LR 參考」對照圖與量測，並判定是否達標。
cases.json 格式：[{"name": "吧檯", "input": "原圖資料夾", "reference": "LR成品.jpg 或省略"}, ...]
無 LR 參考時只檢查場景健康度（室內過曝、暗部、窗外、色偏、飽和）。"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import cv2
from .imageio import list_images, load_image, luma, save_jpg
from .pipeline import run
from .calibrate import register

# 達標門檻
LIMITS = dict(interior_clip=0.02, neutral_a=3.0, neutral_b=6.0, sat_max=0.18,
              window_clip=0.35, dE=3.5, p50_diff=0.04, sat_diff=0.03)


def _lab(x):
    return cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def _sat(x):
    return cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)[..., 1]


def window_mask(originals: list[np.ndarray], shape) -> np.ndarray:
    """窗外（遠比室內亮處）：最暗那張原圖中仍明顯偏亮的區域。"""
    dark = min(originals, key=lambda v: float(np.mean(luma(v))))
    y = luma(cv2.resize(dark, (shape[1], shape[0]), interpolation=cv2.INTER_AREA))
    m = (y > max(0.25, 6 * float(np.median(y)))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.dilate(m, np.ones((9, 9), np.uint8)).astype(bool)


def scene_metrics(out: np.ndarray, originals: list[np.ndarray]) -> dict:
    Y = luma(out)
    win = window_mask(originals, out.shape)
    inner = ~win
    lab, sat = _lab(out), _sat(out)
    neutral = inner & (sat < 0.12) & (Y > 0.55) & (Y < 0.97)
    mid = min(originals, key=lambda v: abs(float(np.median(luma(v))) - 0.35))
    ym = luma(cv2.resize(mid, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_AREA))
    darkreg = inner & (ym < np.percentile(ym[inner], 10))
    return dict(
        p5=float(np.percentile(Y[inner], 5)), p50=float(np.median(Y[inner])),
        p90=float(np.percentile(Y[inner], 90)),
        interior_clip=float((Y[inner] > 0.98).mean()),
        dark_median=float(np.median(Y[darkreg])),
        window_frac=float(win.mean()),
        window_clip=float((Y[win] > 0.98).mean()) if win.any() else 0.0,
        window_detail=float(np.std(Y[win])) if win.any() else 0.0,
        neutral_a=float(lab[..., 1][neutral].mean()) if neutral.any() else 0.0,
        neutral_b=float(lab[..., 2][neutral].mean()) if neutral.any() else 0.0,
        sat=float(sat[inner].mean()))


def ref_metrics(out: np.ndarray, ref: np.ndarray) -> tuple[dict, np.ndarray]:
    ref_w, m, n_in = register(out, ref)
    d = _lab(out)[m] - _lab(ref_w)[m]
    Yo, Yr = luma(out)[m], luma(ref_w)[m]
    so, sr = _sat(out)[m], _sat(ref_w)[m]
    return dict(inliers=n_in, overlap=float(m.mean()),
                dE=float(np.sqrt((d ** 2).sum(-1)).mean()),
                dL=float(d[:, 0].mean()), da=float(d[:, 1].mean()), db=float(d[:, 2].mean()),
                p5_diff=float(np.percentile(Yo, 5) - np.percentile(Yr, 5)),
                p50_diff=float(np.median(Yo) - np.median(Yr)),
                p95_diff=float(np.percentile(Yo, 95) - np.percentile(Yr, 95)),
                sat_diff=float(so.mean() - sr.mean())), ref_w * m[..., None]


def _judge(s: dict, r: dict | None) -> list[str]:
    L, bad = LIMITS, []
    if s["interior_clip"] > L["interior_clip"]: bad.append(f"室內過曝 {s['interior_clip']:.1%}")
    if abs(s["neutral_a"]) > L["neutral_a"]: bad.append(f"白牆偏{'洋紅' if s['neutral_a'] > 0 else '綠'} a*={s['neutral_a']:.1f}")
    if abs(s["neutral_b"]) > L["neutral_b"]: bad.append(f"白牆偏{'黃' if s['neutral_b'] > 0 else '藍'} b*={s['neutral_b']:.1f}")
    if s["sat"] > L["sat_max"]: bad.append(f"過度飽和 {s['sat']:.3f}")
    if s["window_frac"] > 0.005 and s["window_clip"] > L["window_clip"]: bad.append(f"窗外過曝 {s['window_clip']:.0%}")
    if r:
        if r["dE"] > L["dE"]: bad.append(f"與 LR 色差 ΔE={r['dE']:.2f}")
        if abs(r["p50_diff"]) > L["p50_diff"]: bad.append(f"與 LR 亮度差 {r['p50_diff']:+.3f}")
        if abs(r["sat_diff"]) > L["sat_diff"]: bad.append(f"與 LR 飽和差 {r['sat_diff']:+.3f}")
    return bad


def _tile(x, h=360):
    s = h / x.shape[0]
    t = cv2.resize(np.clip(x, 0, 1), (round(x.shape[1] * s), h), interpolation=cv2.INTER_AREA)
    return t


def _label(t, text):
    t = (t * 255).astype(np.uint8).copy()
    cv2.putText(t, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return t.astype(np.float32) / 255


def verify(cases_path, out_dir, log=print) -> dict:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    base = Path(cases_path).resolve().parent
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for c in cases:
        name = c["name"]
        inp = (base / c["input"]).resolve()
        rep = run(inp, out / name, previews=False, log=lambda *_: None)
        res = {}
        for g in rep["groups"]:
            fn = out / name / g["outputs"][0]["file"]
            app = load_image(fn)
            originals = [load_image(inp / f, 1024) for f in g["files"]]
            s = scene_metrics(load_image(fn, 1024), originals)
            r, ref_w = None, None
            if c.get("reference"):
                r, ref_w = ref_metrics(load_image(fn, 1024), load_image((base / c["reference"]).resolve(), 1024))
            bad = _judge(s, r)
            row = [_label(_tile(v), f) for v, f in zip(originals, g["files"])]
            row.append(_label(_tile(app), "APP"))
            if ref_w is not None:
                row.append(_label(_tile(ref_w), "LR"))
            save_jpg(out / f"對照_{name}_{g['name']}.jpg", np.hstack(row), 88)
            res[g["name"]] = dict(scene=s, reference=r, auto_ev=g["outputs"][0]["auto_ev"],
                                  passed=not bad, problems=bad)
            log(f"[{name}/{g['name']}] {'達標' if not bad else '未達標：' + '；'.join(bad)}")
        summary[name] = res
    (out / "驗收.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary
