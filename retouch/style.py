"""一鍵風格：以 xmp 數值近似 Lightroom 的基本調整（非 Adobe 演算法，為近似值）。
輸入為線性輻射量（HDR，可 >1），輸出 sRGB 編碼 [0,1]。
順序：自動白平衡（部分中性化）→ 白平衡增量 → 曝光（xmp + 自動）→ 亮部／陰影（大半徑平滑底層的局部增益）
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
    base_gamma=1.0,    # 中間調 gamma（<1 提亮中間調），以成品校準
    vibrance=1.0, saturation=1.0, hsl_sat=1.0, hsl_lum=0.35, hsl_hue=30.0,
    hdr_knee=1.0,      # 底層亮度超過「室內第 90 百分位 × 此倍數」才壓縮（窗外等 HDR 亮部）
    hdr_slope=0.3,     # 壓縮後 log 斜率（越小壓越多）
    hdr_cap=0.2,       # 底層超出 knee 後最多保留幾 EV
    bright_sat=0.0,    # 壓暗的窗景區彩度補回比例
    bright_dim=0.0,    # 窗戶與窗邊亮區局部壓暗 EV（自動曝光不計入，只影響亮區）
    hdr_dehaze=0.0,    # 窗外去霧比例（扣除窗景霧白底），0＝不處理
    hdr_pull=0.0,      # 窗外（超出 knee 2 EV 以上）額外壓暗的 EV，讓窗景略暗於白牆、保有顏色
    clip_max=0.03,     # 自動曝光：室內過曝像素比例上限（LR 成品約 2.5%）
    white_b=0.0,       # 白牆、天花最終色調（Lab b*）；0＝不額外調整（依 2-4：亮部保留米白）
    wb_shadow=1.0,     # 中暗部白平衡修正比例（1＝與亮面相同，與 2-4 一致：中暗部偏冷）
    chroma_smooth=0.003,  # 色度平滑半徑（佔寬度比例），去 JPG 色斑
    neutral_keep=1.0,  # 低彩度面殘留偏色保留比例（1＝不處理；2-4 的米白需保留）
    neutral_c_lo=2.0, neutral_c_hi=9.0,  # Lab 彩度：低於 lo 視為中性面、高於 hi 視為有色材質
    split_hi_b=3.0, split_mid_b=-0.5, split_lo_b=-1.0, split_lo_a=0.0,  # 分離色調（Lab）：亮部 b*、中暗部 b*／a* 偏移，依 2-4 量得
    sharpen=0.45, sharpen_radius=1.0,  # 輸出銳化量與半徑（2000px 寬時的像素）
    texture_keep=0.0,  # 中尺度紋理補回比例（0＝不補；補回會在牆面產生波紋與光暈，改以減弱局部調整保留紋理）
    texture_cap=0.10,  # 每像素補回上限（log 亮度），避免邊緣光暈
    shoulder=0.85,     # 高光肩部起點（輸出亮度）
    white_max=0.97,    # 高光肩部上限：最亮處趨近此值，不硬切成純白
    single_ev_max=2.5, # 單張照片自動提亮上限（EV）
    clip_src_max=0.15, # 來源剪裁比例超過此值（大面積過曝）時，自動曝光不提亮；窗外等小面積剪裁不影響提亮
    wb_auto=0.5,       # 自動白平衡強度：0＝相機白平衡，1＝白牆完全中性（保留部分暖光氛圍）
)
# 若有以成品校準過的常數（python -m retouch calibrate-style），覆寫預設值
from . import PRESETS as _PRESETS
_K_FILE = _PRESETS / "style_k.json"
if _K_FILE.exists():
    import json
    K.update(json.loads(_K_FILE.read_text(encoding="utf-8"))["K"])
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


def _midband(lY: np.ndarray) -> np.ndarray:
    """中尺度紋理帶（約寬度 0.3%～2.5%）：牆面雲紋、櫃體斑駁。大半徑模糊在縮圖上做以加速。"""
    h, w = lY.shape
    s1, s2 = 0.003 * w, 0.025 * w
    sc = min(1.0, 1000 / w)
    small = cv2.resize(lY, (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=cv2.INTER_AREA) if sc < 1 else lY
    b2 = cv2.GaussianBlur(small, (0, 0), s2 * sc)
    if sc < 1:
        b2 = cv2.resize(b2, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(lY, (0, 0), s1) - b2


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


def hdr_compress(x: np.ndarray, knee: float, slope: float, radius_frac: float, cap: float = None,
                 pull: float = 0.0, dehaze: float = 0.0) -> np.ndarray:
    """HDR 亮部壓縮：只壓 log 亮度的大尺度底層、保留細節（窗外不再整片過曝）。"""
    Y = np.maximum(luma(x), 1e-6)
    lY = np.log2(Y)
    base = _guided_base(lY, radius_frac / 2, eps=0.1)
    # knee 相對於場景：室內亮面（第 90 百分位）的 knee 倍，只壓遠比室內亮的窗外
    t = np.log2(knee) + float(np.percentile(lY[::4, ::4], 90))
    over = np.maximum(base - t, 0)
    # 漸近壓縮：超出 knee 的 EV 數最多被壓到 cap，再亮的窗外也不會整片過曝
    cap = K["hdr_cap"] if cap is None else cap
    # pull：遠比室內亮的窗外再多壓暗 pull EV（超出 4.5 EV 以上才全量），讓窗景比白牆略暗、保有顏色，室內不受影響
    gain = np.exp2(-(over - over / (1 + over * (1 - slope) / cap)))
    if pull > 0:
        # 逐像素判斷（比室內亮面高 3 EV 以上才算窗外，日照沙發不受影響），不用大尺度底層，避免光暈
        # 窗外遮罩外擴幾個像素再羽化：窗框與窗景交界的混合像素一起壓，不留亮邊
        px = max(1, round(x.shape[1] / 700))
        m = _smoothstep(3.0, 4.5, lY - t).astype(np.float32)
        m = cv2.GaussianBlur(cv2.dilate(m, np.ones((2 * px + 1,) * 2, np.uint8)), (0, 0), px)
        gain = gain * np.exp2(-pull * m)
        y = x * gain[..., None]
        if dehaze > 0 and (m > 0.5).any():
            # 窗外去霧：扣掉窗景的霧白底（窗內第 5 百分位亮度），恢復顏色與層次，避免壓暗後變灰
            haze = float(np.percentile(luma(y)[m > 0.5], 5))
            y = np.maximum(y - (dehaze * haze) * m[..., None], 0)
        return y.astype(np.float32)
    return (x * gain[..., None]).astype(np.float32)


def auto_wb(R: np.ndarray, strength: float, clipped=None) -> np.ndarray:
    """由亮面中最不飽和的像素（白牆、天花）估計色偏，回傳部分中性化的 RGB 增益。
    剪裁像素（純白，不代表真實顏色）不列入估計。"""
    if strength <= 0:
        return np.ones(3, np.float32)
    x = R[::8, ::8].reshape(-1, 3).astype(np.float32)
    if clipped is not None:
        x = x[clipped[::8, ::8].ravel() < 0.05]
    y = luma(x)
    lo, hi = np.percentile(y, [50, 98])
    x = x[(y > lo) & (y < hi)]
    if len(x) < 50:
        return np.ones(3, np.float32)
    mx = x.max(1)
    chroma = (mx - x.min(1)) / np.maximum(mx, 1e-6)
    x = x[chroma <= np.percentile(chroma, 30)]
    m = np.median(x, 0)
    g = (m @ np.array([0.2126, 0.7152, 0.0722], np.float32)) / np.maximum(m, 1e-6)
    g = g ** strength
    return (g / (g @ np.array([0.2126, 0.7152, 0.0722], np.float32))).astype(np.float32)


def scene_warmth(R: np.ndarray) -> float:
    """場景暖度 0–1：亮面平均 b*（黃）越高越暖。用來決定預設集的色溫／色調增量要套多少：
    偏黃的照片完整套用，本來就中性的照片不套（否則中性白沙發會被染成淡紫）。"""
    x = np.clip(linear_to_srgb(R[::8, ::8] / max(np.percentile(luma(R[::8, ::8]), 95), 1e-6) * 0.9), 0, 1)
    lab = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    m = (L > 55) & (np.hypot(A, B) < 25)
    if m.sum() < 50:
        return 1.0
    return float(np.clip((np.median(B[m]) - 3.0) / 8.0, 0, 1))


def wb_gains(p: dict, k: dict = K, warmth: float = 1.0) -> np.ndarray:
    t, n = p.get("IncrementalTemperature", 0) * warmth, p.get("IncrementalTint", 0) * warmth
    g = np.array([2 ** (k["temp"] * t), 2 ** (-k["tint"] * n), 2 ** (-k["temp"] * t)], np.float32)
    return g / (g @ np.array([0.2126, 0.7152, 0.0722], np.float32))


def apply_style(R: np.ndarray, p: dict, ev: float = 0.0, k: dict = K, clipped=None, hdr: bool = True) -> np.ndarray:
    """clipped：來源已剪裁（所有曝光都過曝）的區域權重 0–1。該處沒有色彩資訊，維持中性白，
    不套白平衡增益（否則純白會被染成紫藍），也不壓暗（壓暗只會變成一片灰，不是真的恢復）。
    hdr：輸入是否為多張合成的 HDR。單張 JPG 本身已到白點，不做 HDR 亮部壓縮（否則畫面發灰、失去層次）。"""
    x = R.astype(np.float32) * np.float32(2 ** (p.get("Exposure2012", 0) + ev))
    # 白平衡隨亮度：亮面（白牆、天花）完整修正；暗部（地板、石材、深色家具）只部分修正，
    # 避免整張一起往藍調、暗部變紫，造成亮暗面色調不一致（看起來顏色雜）
    G = auto_wb(R, k["wb_auto"], clipped) * wb_gains(p, k, scene_warmth(R))
    yw = np.clip(linear_to_srgb(luma(x)), 0, 1)
    tw = np.clip((yw - 0.40) / 0.35, 0, 1)
    wgt = k["wb_shadow"] + (1 - k["wb_shadow"]) * (tw * tw * (3 - 2 * tw))
    x = x * np.exp(np.log(G)[None, None, :] * wgt[..., None])
    if clipped is not None:
        Yx = luma(x)[..., None]
        x = x + (Yx - x) * clipped[..., None]
    tex_in = _midband(np.log(luma(x) + 1e-4)) if k["texture_keep"] > 0 else None
    # 亮部／陰影：依平滑底層決定局部 EV 增益，保留細節
    yb = np.clip(linear_to_srgb(luma(x)), 0, 1.5)
    base = _guided_base(yb, k["base_radius"])
    ws = 1 - _smoothstep(0.05, 0.6, base)
    wh = _smoothstep(0.45, 1.05, base)
    loc = p.get("Shadows2012", 0) / 100 * k["shadows"] * ws + p.get("Highlights2012", 0) / 100 * k["highlights"] * wh
    if clipped is not None:
        loc = np.where(loc < 0, loc * (1 - clipped), loc)
    x *= np.exp2(loc)[..., None]
    if k["bright_dim"] > 0:
        # 窗景局部壓暗：逐像素以壓縮前 log 亮度判斷（比室內亮面高 4.5～6 EV 以上才是窗外；
        # 日照的沙發、桌面約高 1～4 EV，不受影響，避免出現灰藍色塊），室內中暗部不受影響
        lY = np.log2(np.maximum(luma(x), 1e-6))
        dw = _smoothstep(4.5, 6.0, lY - float(np.percentile(lY[::4, ::4], 90))).astype(np.float32)
        # 外擴再羽化：窗框、欄杆等細黑邊旁的混合像素一起壓，不留白邊
        px = max(2, round(x.shape[1] / 300))
        dw = cv2.GaussianBlur(cv2.dilate(dw, np.ones((2 * px + 1,) * 2, np.uint8)), (0, 0), px / 2)
        dim = k["bright_dim"] * dw
        # 只有窗景本身補色（窗框外擴區不補）
        winm = cv2.GaussianBlur(_smoothstep(4.5, 6.0, lY - float(np.percentile(lY[::4, ::4], 90))).astype(np.float32),
                              (0, 0), max(1.0, x.shape[1] / 1500))
    else:
        dim = None
    if hdr:
        x = hdr_compress(x, k["hdr_knee"], k["hdr_slope"], k["base_radius"], k["hdr_cap"], k["hdr_pull"], k["hdr_dehaze"])
    g = linear_to_srgb(x)
    # 亮度曲線：白色、黑色、對比（以亮度比例套用，維持色相）
    Y = luma(g)
    Yc = np.clip(Y, 0, None)
    w, b, c = p.get("Whites2012", 0) / 100, p.get("Blacks2012", 0) / 100, p.get("Contrast2012", 0) / 100
    # 基礎影調（對應 LR 相機描述檔的底曲線）：中間調 gamma，端點不動
    Yc = np.where(Yc < 1, np.clip(Yc, 0, 1) ** k["base_gamma"], Yc)
    T = Yc * (1 + k["whites"] * w * np.clip(Yc, 0, 1.2) ** 3)
    T = T * (1 + k["blacks"] * b * (1 - np.clip(T, 0, 1)) ** 4)
    t = np.clip(T, 0, 1)
    T = np.where(T <= 1, t - k["contrast"] * c * np.sin(2 * np.pi * t) / (2 * np.pi), T)
    g = g * (T / np.maximum(Y, 1e-5))[..., None]
    # 高光肩部：亮度超過 shoulder 後平滑趨近 white_max，白櫃、白牆、天花保留邊緣與漸層，不變成一片死白
    Yg = luma(g)
    s0, wm = k["shoulder"], k["white_max"]
    Ys = np.where(Yg > s0, s0 + (wm - s0) * np.tanh((Yg - s0) / (wm - s0)), Yg)
    g = g * (Ys / np.maximum(Yg, 1e-5))[..., None]
    if tex_in is not None:
        # 補回局部亮部／陰影調整壓掉的中尺度紋理（牆面、水泥櫃的雲紋），最亮與最暗處減量避免光暈
        Yg = np.maximum(luma(g), 1e-4)
        lost = np.clip(tex_in - _midband(np.log(Yg)), -k["texture_cap"], k["texture_cap"])
        guard = np.clip(Yg / 0.15, 0, 1) * np.clip((0.98 - Yg) / 0.10, 0, 1)
        g = g * np.exp(k["texture_keep"] * lost * guard)[..., None]
    if dim is not None:
        # 在輸出影調後才壓（壓在線性空間會被基礎影調拉回、且通道先被剪裁而變灰），保留窗景顏色
        g = g * np.exp2(-0.5 * dim)[..., None]
        # 窗景壓暗後補回色彩（HDR 壓縮會讓窗景變淡，只壓暗會變灰）
        Yd = luma(g)[..., None]
        g = Yd + (g - Yd) * (1 + k["bright_sat"] * winm)[..., None]
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
    return sharpen(color_cleanup(np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1), k), k)


def sharpen(rgb: np.ndarray, k: dict = K) -> np.ndarray:
    """輸出銳化（類似 LR 輸出銳化）：只作用在亮度，半徑依影像寬度縮放，平坦區（雜訊）減量。"""
    amt = k["sharpen"]
    if amt <= 0:
        return rgb
    r = k["sharpen_radius"] * rgb.shape[1] / 2000
    Y = luma(rgb)
    d = Y - cv2.GaussianBlur(Y, (0, 0), max(0.5, r))
    edge = np.clip(np.abs(cv2.GaussianBlur(d, (0, 0), 1.0)) / 0.01, 0, 1)  # 平坦牆面的細雜訊少銳化
    Yn = Y + amt * d * (0.35 + 0.65 * edge)
    return np.clip(rgb * (np.maximum(Yn, 0) / np.maximum(Y, 1e-4))[..., None], 0, 1).astype(np.float32)


def _guided(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    box = lambda a: cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def color_cleanup(rgb: np.ndarray, k: dict = K) -> np.ndarray:
    """色彩淨化：(1) 色度沿亮度邊緣平滑，去除 JPG 色斑；(2) 低彩度面（白牆、天花、灰櫃）的殘留偏色降低，
    有彩度的材質（木皮、石材紋、燈罩）不受影響。"""
    if k["neutral_keep"] >= 1 and k["chroma_smooth"] <= 0:
        return rgb
    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)
    L, A, B = lab[..., 0] / 100, lab[..., 1], lab[..., 2]
    if k["chroma_smooth"] > 0:
        r = max(2, round(k["chroma_smooth"] * rgb.shape[1]))
        A, B = _guided(L, A, r, 1e-3), _guided(L, B, r, 1e-3)
    C = np.hypot(A, B)
    t = np.clip((C - k["neutral_c_lo"]) / (k["neutral_c_hi"] - k["neutral_c_lo"]), 0, 1)
    f = k["neutral_keep"] + (1 - k["neutral_keep"]) * (t * t * (3 - 2 * t))
    A, B = A * f, B * f
    # 公司風格白：亮且接近中性的面（白牆、天花）統一調到 white_b，木皮、石材等有彩度的材質不動
    if k.get("white_b", 0):
        tl = np.clip((L - 0.55) / 0.30, 0, 1)
        C2 = np.hypot(A, B)  # 淨化後彩度；略帶暖色的白牆（C≈4–8）也納入，明顯有色材質（C>14）不動
        t2 = np.clip((C2 - 6.0) / 8.0, 0, 1)
        wgt = (tl * tl * (3 - 2 * tl)) * (1 - (t2 * t2 * (3 - 2 * t2)))
        B = B + (k["white_b"] - B) * wgt * 0.85
    # 分離色調（依 2-4）：亮部偏米白、中暗部偏冷帶洋紅
    ss = lambda a, b_, x: (lambda t: t * t * (3 - 2 * t))(np.clip((x - a) / (b_ - a), 0, 1))
    w_hi = ss(0.84, 0.94, L)          # 最亮處（白牆、天花）
    w_lo = 1 - ss(0.62, 0.80, L)      # 中暗部（地板、石材、櫃體陰影）
    B = B + k["split_hi_b"] * w_hi + k["split_lo_b"] * w_lo + k["split_mid_b"] * (1 - w_hi) * (1 - w_lo)
    A = A + k["split_lo_a"] * w_lo
    lab[..., 1], lab[..., 2] = A, B
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)


# ---------- 自動曝光 ----------
# 以亮度第 90 百分位（白牆、天花等亮面）為準，不用中位數：中位數受深色家具／牆面比例影響太大。
TARGET_PCT = 90
DEFAULT_TARGET = 0.91  # 由 2-4.jpg 學得；可用 learn-target 由其他成品重學


def exposure_metric(rgb: np.ndarray) -> float:
    return float(np.percentile(luma(rgb), TARGET_PCT))


def learn_target(paths: list[str | Path]) -> float:
    """由使用者的成品 JPG 學習目標亮度（第 90 百分位）。"""
    return float(np.median([exposure_metric(load_image(p, 1024)) for p in paths]))


def _bisect(f, target, lo, hi, iters):
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def auto_exposure(R: np.ndarray, p: dict, target: float = DEFAULT_TARGET, k: dict = K,
                  lo: float = -6.0, hi: float = 6.0, iters: int = 20, clipped=None, hdr: bool = True) -> float:
    """依場景動態決定 ev：亮面（第 90 百分位）對到 target，
    但過曝像素比例不得超過 clip_max（白牆、白沙發不過曝）；兩者取較暗者。R 建議先縮到 ~768px。
    clipped：來源已剪裁區域。剪裁比例 > clip_src_max 時不再提亮（提亮只會擴大死白，無法恢復）。"""
    cache = {}
    # 曝光依「窗外未額外壓暗」的畫面決定：窗外壓暗不應讓室內跟著被提亮
    k = {**k, "hdr_pull": 0.0, "bright_dim": 0.0}
    # 室內遮罩：排除遠比室內亮的窗外（平滑後輻射量 > 室內第 90 百分位的 3 倍）與來源剪裁區
    Yr = cv2.GaussianBlur(luma(R), (0, 0), 3)
    inner = Yr < 3 * np.percentile(Yr, 90)
    if clipped is not None:
        inner &= clipped < 0.5
    src_clip = float((clipped > 0.5).mean()) if clipped is not None else 0.0
    if src_clip > k["clip_src_max"]:
        hi = min(hi, 0.0)
    if not hdr:  # 單張 JPG：提亮過多只會放大雜訊與色斑
        hi = min(hi, k["single_ev_max"])

    def out(ev):
        if ev not in cache:
            cache[ev] = luma(apply_style(R, p, ev, k, clipped, hdr))
        return cache[ev]

    ev_t = _bisect(lambda e: float(np.percentile(out(e), TARGET_PCT)), target, lo, hi, iters)
    clip = lambda e: float((out(e)[inner] > 0.98).mean())
    if clip(ev_t) <= k["clip_max"]:
        return ev_t
    return _bisect(clip, k["clip_max"], lo, ev_t, iters)
