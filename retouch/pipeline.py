"""批次流程：分群 → 對齊 → 校準 → 光狀態 → HDR／燈光層 → 自動曝光 → 套風格 → 輸出。
分析在縮圖上做（預設長邊 1536），合成在全尺寸逐張串流，避免 45MP × 多張吃光記憶體。"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import cv2
from .imageio import list_images, load_image, resize_max, save_jpg, save_tiff16
from .grouping import group_by_angle
from .align import chain_align, apply_warp
from .merge import (calibrate_chain, difference_matrix, dilate_masks, cluster_states, hdr_merge,
                    light_layer, clip_mask, dark_mask, srgb_curve, States)
from .style import apply_style, auto_exposure, DEFAULT_TARGET
from .xmp import parse_xmp

from . import PRESETS


def default_target() -> float:
    f = PRESETS / "target.json"
    if f.exists():
        return float(json.loads(f.read_text(encoding="utf-8"))["target"])
    return DEFAULT_TARGET


def _fit_shape(v: np.ndarray, shape) -> np.ndarray:
    h, w = shape[:2]
    return v if v.shape[:2] == (h, w) else cv2.resize(v, (w, h), interpolation=cv2.INTER_AREA)


def exposure_label(v: np.ndarray) -> tuple[str, float, float]:
    c, d = float(clip_mask(v).mean()), float(dark_mask(v).mean())
    lab = "過曝" if c > 0.10 else "過暗" if d > 0.30 else "正常"
    return lab, c, d


def analyze_group(thumbs: list[np.ndarray], state_thr: float = 0.015) -> dict:
    """單一機位分析（縮圖）。回傳 warps、校準、光狀態等。"""
    shape = thumbs[0].shape
    vs = [_fit_shape(t, shape) for t in thumbs]
    warps, align_rep = chain_align(vs)
    vs = [apply_warp(v, w) for v, w in zip(vs, warps)]
    cal = calibrate_chain(vs)
    D, B, masks = difference_matrix(vs, cal.curves)
    # 第二輪：排除燈光不同的像素重新擬合曲線
    cal = calibrate_chain(vs, exclude=dilate_masks(masks))
    D, B, masks = difference_matrix(vs, cal.curves)
    states = cluster_states(D, B, state_thr)
    return dict(vs=vs, warps=warps, align=align_rep, cal=cal, D=D, B=B, states=states)


def _compose(frames_of, curves, evs, states, switches: dict) -> np.ndarray:
    """環境光 HDR + Σ a_k·Δ_k。frames_of(members) 產生 (i, 已對齊編碼影像)。"""
    R_amb, _ = hdr_merge(frames_of(states.members[states.ambient]), curves, evs)
    for s, (mem, kind) in enumerate(zip(states.members, states.kinds)):
        amt = float(switches.get(f"S{s}", 0))
        if kind == "light" and amt:
            R_s, den = hdr_merge(frames_of(mem), curves, evs)
            R_amb = R_amb + amt * light_layer(R_s, R_amb, den)
    return R_amb


def run(input_dir, out_dir, style_path=PRESETS / "style_default.xmp", lights_path=None,
        target: float | None = None, tiff: bool = False, analysis_side: int = 1536,
        group_thr: float = 0.35, state_thr: float = 0.015, previews: bool = True,
        log=print, progress=None) -> dict:
    """input_dir 可為資料夾，或照片路徑清單（介面逐張匯入時）。progress(完成數, 總數) 供介面顯示進度。"""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(q) for q in input_dir) if isinstance(input_dir, (list, tuple)) else list_images(input_dir)
    if not paths:
        raise FileNotFoundError(f"資料夾內沒有影像：{input_dir}")
    p = parse_xmp(style_path)
    if p["_ignored"]:
        log(f"注意：xmp 內這些項目 APP 不處理（請在 LR 做）：{p['_ignored']}")
    target = default_target() if target is None else target
    lights = json.loads(Path(lights_path).read_text(encoding="utf-8")) if lights_path else {}

    log(f"讀取 {len(paths)} 張縮圖…")
    thumbs = [load_image(q, analysis_side) for q in paths]
    groups, S = group_by_angle(thumbs, group_thr)
    log(f"分成 {len(groups)} 個機位")
    report, template = {"target": target, "groups": []}, {}
    pv_dir = out / "預覽"
    if previews:
        pv_dir.mkdir(exist_ok=True)

    for gi, idx in enumerate(groups):
        if progress:
            progress(gi, len(groups))
        name = f"{gi + 1:02d}_{paths[idx[0]].stem}"
        files = [paths[i].name for i in idx]
        log(f"[{name}] {len(idx)} 張")
        full0 = load_image(paths[idx[0]])
        scale = full0.shape[1] / thumbs[idx[0]].shape[1]
        g = dict(name=name, files=files)

        if len(idx) == 1:
            curves, evs = [srgb_curve()], [0.0]
            states = cluster_states(np.zeros((1, 1)), np.zeros((1, 1)))
            full_warps = [np.eye(2, 3, dtype=np.float32)]
            vs = [thumbs[idx[0]]]
        else:
            A = analyze_group([thumbs[i] for i in idx], state_thr)
            curves, evs, states, vs = A["cal"].curves, A["cal"].ev, A["states"], A["vs"]
            full_warps = []
            for w in A["warps"]:
                w = w.copy(); w[:, 2] *= scale; full_warps.append(w)
            g.update(align=[dict(file=files[r["index"]], dx=round(r["dx"] * scale, 2),
                                 dy=round(r["dy"] * scale, 2), cc=round(r["cc"], 3),
                                 trusted=bool(r.get("trusted", True))) for r in A["align"]],
                     calibration_root=files[A["cal"].root],
                     calibration_tree=[[files[i], files[j]] for i, j in A["cal"].tree],
                     curves_ok=[c.ok for c in curves],
                     difference=np.round(A["D"], 4).tolist())
        g["exposure"] = []
        for k, v in enumerate(vs):
            lab, c, d = exposure_label(v)
            g["exposure"].append(dict(file=files[k], label=lab, ev=round(evs[k], 2),
                                      clipped=round(c, 3), dark=round(d, 3)))
        g["states"] = [dict(id=f"S{s}", kind=kind, files=[files[i] for i in mem])
                       for s, (mem, kind) in enumerate(zip(states.members, states.kinds))]
        template[name] = {f"S{s}": 0 for s, kd in enumerate(states.kinds) if kd == "light"}
        switches = lights.get(name, {})

        def frames_of(members, _idx=idx, _full0=full0, _warps=full_warps):
            for i in members:
                v = _full0 if i == 0 else _fit_shape(load_image(paths[_idx[i]]), _full0.shape)
                yield i, apply_warp(v, _warps[i])

        def thumb_frames(members, _vs=vs):
            for i in members:
                yield i, _vs[i]

        # 主輸出 + 場景變體（如開門／關門）
        outputs = [("", states)]
        for s, kind in enumerate(states.kinds):
            if kind == "variant":
                vst = States(states.members, s, states.kinds)
                outputs.append((f"_變體S{s}", vst))
        g["outputs"] = []
        for suffix, st in outputs:
            R_small = _compose(thumb_frames, curves, evs, st, switches)
            ev = auto_exposure(resize_max(R_small, 768), p, target)
            R = _compose(frames_of, curves, evs, st, switches)
            img = apply_style(R, p, ev)
            fn = out / f"{name}{suffix}.{'tif' if tiff else 'jpg'}"
            (save_tiff16 if tiff else save_jpg)(fn, img)
            g["outputs"].append(dict(file=fn.name, auto_ev=round(ev, 2),
                                     switches={k: v for k, v in switches.items() if v}))
            log(f"  → {fn.name}（自動曝光 {ev:+.2f} EV）")
            del R, img
        if previews:
            R_amb = _compose(thumb_frames, curves, evs, states, {})
            ev = auto_exposure(resize_max(R_amb, 768), p, target)
            save_jpg(pv_dir / f"{name}_環境光.jpg", apply_style(R_amb, p, ev), 90)
            for s, kind in enumerate(states.kinds):
                if kind == "light":
                    R_l = _compose(thumb_frames, curves, evs, states, {f"S{s}": 1})
                    save_jpg(pv_dir / f"{name}_S{s}_燈光.jpg", apply_style(R_l, p, ev), 90)
        report["groups"].append(g)

    if progress:
        progress(len(groups), len(groups))
    (out / "報告.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "燈光開關表_範本.json").write_text(json.dumps(template, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    return report
