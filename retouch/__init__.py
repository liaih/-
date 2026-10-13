"""快速修圖 APP — 第一階段命令列版：分群、對齊、曝光判斷、燈光圖層、套風格、輸出 JPG。"""
from pathlib import Path as _Path
import sys as _sys

# 預設集位置：原始碼執行時為 repo/presets；PyInstaller 打包後為解壓目錄下的 presets
PRESETS = (_Path(_sys._MEIPASS) if getattr(_sys, "frozen", False) else _Path(__file__).resolve().parent.parent) / "presets"
