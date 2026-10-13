"""命令列：
  python -m retouch run 輸入資料夾 -o 輸出資料夾 [--lights 燈光開關表.json] [--tiff]
  python -m retouch learn-target 成品1.jpg [成品2.jpg ...]
  python -m retouch calibrate-style 原圖資料夾 成品.jpg [原圖資料夾2 成品2.jpg ...]
  python -m retouch verify cases.json -o 驗收資料夾
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .pipeline import run, PRESETS
from .style import learn_target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="retouch", description="快速修圖：批次分群、HDR、燈光圖層、套風格")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="批次處理資料夾")
    r.add_argument("input")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--style", default=str(PRESETS / "style_default.xmp"), help="LR 預設集 .xmp")
    r.add_argument("--lights", help="燈光開關表 JSON（先跑一次會產生範本）")
    r.add_argument("--target", type=float, help="套風格後亮度第 90 百分位（0–1），預設用 learn-target 結果")
    r.add_argument("--tiff", action="store_true", help="輸出 16-bit TIFF（預設 JPG 品質 95）")
    r.add_argument("--analysis-side", type=int, default=1536, help="分析縮圖長邊")
    r.add_argument("--group-thr", type=float, default=0.35, help="機位分群相似度閾值")
    r.add_argument("--state-thr", type=float, default=0.015, help="光狀態差異比例閾值")
    r.add_argument("--no-previews", action="store_true", help="不輸出環境光／各燈光預覽")
    t = sub.add_parser("learn-target", help="由成品 JPG 學習目標亮度")
    t.add_argument("samples", nargs="+")
    c = sub.add_parser("calibrate-style", help="以 LR 成品校準風格常數（可多組）")
    c.add_argument("pairs", nargs="+", help="原圖資料夾 成品.jpg [原圖資料夾2 成品2.jpg ...]")
    c.add_argument("--style", default=str(PRESETS / "style_default.xmp"))
    v = sub.add_parser("verify", help="自動化驗收：原圖｜APP｜LR 對照與量測")
    v.add_argument("cases", help="cases.json")
    v.add_argument("-o", "--output", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        from .verify import verify
        res = verify(a.cases, a.output)
        return 0 if all(g["passed"] for c in res.values() for g in c.values()) else 1
    if a.cmd == "calibrate-style":
        from .calibrate import calibrate_style
        if len(a.pairs) % 2:
            ap.error("calibrate-style 需成對提供：原圖資料夾 成品.jpg")
        r = calibrate_style(list(zip(a.pairs[::2], a.pairs[1::2])), a.style)
        print("已存 presets/style_k.json：", r["K"])
        return 0
    if a.cmd == "learn-target":
        v = learn_target(a.samples)
        (PRESETS / "target.json").write_text(json.dumps({"target": round(v, 4), "samples": [Path(x).name for x in a.samples]},
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"目標亮度（第 90 百分位）= {v:.4f}（已存 presets/target.json）")
        return 0
    run(a.input, a.output, a.style, a.lights, a.target, a.tiff, a.analysis_side,
        a.group_thr, a.state_thr, not a.no_previews)
    return 0


if __name__ == "__main__":
    sys.exit(main())
