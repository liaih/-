"""解析 Lightroom / Camera Raw 預設集（.xmp）的 crs: 數值。支援屬性與子元素兩種寫法。"""
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET

CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
HSL_COLORS = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]

# APP 會用到的欄位與預設值（其餘忽略：遮色片、幾何、銳利化等留在 LR）
FIELDS = {
    "IncrementalTemperature": 0.0, "IncrementalTint": 0.0,
    "Exposure2012": 0.0, "Contrast2012": 0.0, "Highlights2012": 0.0, "Shadows2012": 0.0,
    "Whites2012": 0.0, "Blacks2012": 0.0, "Vibrance": 0.0, "Saturation": 0.0,
    **{f"{k}Adjustment{c}": 0.0 for k in ("Hue", "Saturation", "Luminance") for c in HSL_COLORS},
}
# 解析到但 APP 不處理、需提醒使用者的欄位（非 0 時）
UNSUPPORTED = ["Texture", "Clarity2012", "Dehaze", "Sharpness", "LuminanceSmoothing",
               "ColorNoiseReduction", "PostCropVignetteAmount", "GrainAmount",
               "ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights"]


def _num(s: str) -> float | None:
    try:
        return float(s.strip().replace("+", ""))
    except (ValueError, AttributeError):
        return None


def parse_xmp(path: str | Path) -> dict:
    root = ET.fromstring(Path(path).read_bytes())
    raw: dict[str, str] = {}
    for el in root.iter():
        for k, v in el.attrib.items():
            if k.startswith("{" + CRS + "}"):
                raw[k[len(CRS) + 2:]] = v
        if el.tag.startswith("{" + CRS + "}") and el.text and el.text.strip():
            raw.setdefault(el.tag[len(CRS) + 2:], el.text)
    params = dict(FIELDS)
    for k in FIELDS:
        if k in raw and (x := _num(raw[k])) is not None:
            params[k] = x
    params["_ignored"] = {k: _num(raw[k]) for k in UNSUPPORTED if k in raw and _num(raw[k])}
    params["_white_balance"] = raw.get("WhiteBalance", "")
    return params
