"""多段曝光校準、光狀態判斷、HDR 合成。

流程（同一機位、已對齊）：
1. calibrate_chain：亮度轉換曲線校準。以最大生成樹逐級串接（權重 = 共同有效面積），
   每張只對已校準的鄰居擬合，一路接回基準 → 最暗的幾張不必直接對基準。
   第二輪排除第一輪判定「燈光不同」的像素再擬合一次，避免燈光外溢帶偏曲線。
2. difference_matrix：light_difference 兩兩比較 → D（差異比例）、B（i 比 j 亮的比例）。
3. cluster_states：D 做 complete-linkage 閾值聚類 → 光狀態；最暗者為環境光；
   其餘若「單向變亮」為燈光層，否則為場景變體（例：開門／關門）。
4. hdr_merge：hat 權重輻射量合成；燈光層 Δ = max(R_k − R_amb, 0)。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2
from .imageio import luma, srgb_to_linear

LO, HI = 0.05, 0.95
NBINS = 64
LOG_EPS = 1e-6


# ---------- 有效區 ----------
def maxch(v: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(v[..., 0], v[..., 1]), v[..., 2])


def valid_mask(v: np.ndarray, lo: float = LO, hi: float = HI) -> np.ndarray:
    return (maxch(v) < hi) & (luma(v) > lo)


def clip_mask(v: np.ndarray, hi: float = HI) -> np.ndarray:
    return maxch(v) >= hi


def dark_mask(v: np.ndarray, lo: float = LO) -> np.ndarray:
    return luma(v) <= lo


# ---------- 亮度轉換曲線 ----------
@dataclass
class Curve:
    """編碼值 v ∈ [0,1] → 相對輻射量（以基準張為尺度）。內部存 4096 點 LUT。"""
    lut: np.ndarray
    ok: bool = True
    n_pairs: int = 0

    def __call__(self, v: np.ndarray) -> np.ndarray:
        x = np.clip(v, 0, 1) * (len(self.lut) - 1)
        i = np.minimum(x.astype(np.int32), len(self.lut) - 2)
        t = (x - i).astype(np.float32)
        return self.lut[i] * (1 - t) + self.lut[i + 1] * t


def srgb_curve(size: int = 4096) -> Curve:
    v = np.linspace(0, 1, size, dtype=np.float32)
    return Curve(srgb_to_linear(v))


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """加權單調遞增回歸（pool adjacent violators）。"""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w2 = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w2
            c2 = cnt[-2] + cnt[-1]
            vals[-2:] = [v2]; wts[-2:] = [w2]; cnt[-2:] = [c2]
    return np.repeat(vals, cnt)


def _mode(x: np.ndarray, width: float = 0.02) -> float:
    """穩健眾數：直方圖平滑取峰值，再取峰值附近樣本中位數。
    燈光只會讓像素變亮（單側污染），眾數鎖定未受影響的主峰，中位數則會被拉偏。"""
    if x.size < 200:
        return float(np.median(x))
    lo, hi = np.percentile(x, [1, 99])
    if hi - lo < 4 * width:
        return float(np.median(x))
    nb = int(min(400, (hi - lo) / (width / 2)))
    h, e = np.histogram(x, nb, (lo, hi))
    h = cv2.GaussianBlur(h.astype(np.float32).reshape(1, -1), (0, 0), 2).ravel()
    c = (e[np.argmax(h)] + e[np.argmax(h) + 1]) / 2
    near = x[np.abs(x - c) < 1.5 * width]
    return float(np.median(near)) if near.size else float(c)


def _bin_medians(v, lr, ok, min_count, min_frac):
    """單一夥伴：逐 bin 的 (中位數 v, 眾數 log R, 有效數)。
    夥伴在該 bin 有效比例 < min_frac 視為截尾（接近過曝／過暗邊界），整個 bin 不採用，避免偏差。"""
    edges = np.linspace(0, 1, NBINS + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, NBINS - 1)
    tot = np.bincount(idx, minlength=NBINS)
    out = np.full((NBINS, 3), np.nan)
    order = np.argsort(idx[ok], kind="stable")
    io, vo, lo_ = idx[ok][order], v[ok][order], lr[ok][order]
    bounds = np.searchsorted(io, np.arange(NBINS + 1))
    for b in range(NBINS):
        s, e = bounds[b], bounds[b + 1]
        if e - s >= min_count and (e - s) >= min_frac * tot[b]:
            out[b] = (np.median(vo[s:e]), _mode(lo_[s:e]), e - s)
    return out


def fit_curve(samples, min_count: int = 30, min_frac: float = 0.9, size: int = 4096) -> Curve:
    """samples：[(v_i, R_partner, partner_ok), ...]，每個已校準夥伴一組。
    逐 bin 眾數 log R（多夥伴依樣本數加權）→ PAV 單調 → 兩端 log-log 斜率外推。"""
    stats = []
    n_pairs = 0
    for v, r, ok in samples:
        ok = ok & (r > LOG_EPS)
        n_pairs += int(ok.sum())
        if ok.sum() >= min_count * 4:
            stats.append(_bin_medians(v, np.log(np.maximum(r, LOG_EPS)), ok, min_count, min_frac))
    if not stats:
        return Curve(srgb_curve(size).lut, ok=False, n_pairs=n_pairs)
    st = np.stack(stats)                      # (夥伴, bin, 3)
    w = np.nan_to_num(st[..., 2])
    have = w.sum(0) > 0
    if have.sum() < 4:
        return Curve(srgb_curve(size).lut, ok=False, n_pairs=n_pairs)
    xs = (np.nan_to_num(st[..., 0]) * w).sum(0)[have] / w.sum(0)[have]
    ys = (np.nan_to_num(st[..., 1]) * w).sum(0)[have] / w.sum(0)[have]
    ws = w.sum(0)[have]
    ys = _pav(ys, ws)
    # 兩端斜率（log y 對 log x）外推
    lx = np.log(xs)
    k = min(4, len(xs) - 1)
    s_lo = (ys[k] - ys[0]) / (lx[k] - lx[0])
    s_hi = (ys[-1] - ys[-1 - k]) / (lx[-1] - lx[-1 - k])
    s_lo = float(np.clip(s_lo, 0.8, 4.0)); s_hi = float(np.clip(s_hi, 0.8, 6.0))
    grid = np.linspace(0, 1, size, dtype=np.float64)
    lg = np.log(np.maximum(grid, 1e-4))
    out = np.interp(grid, xs, ys)
    out = np.where(grid < xs[0], ys[0] + s_lo * (lg - lx[0]), out)
    out = np.where(grid > xs[-1], ys[-1] + s_hi * (lg - lx[-1]), out)
    lut = np.exp(out).astype(np.float32)
    lut[0] = 0.0
    return Curve(np.maximum.accumulate(lut), ok=True, n_pairs=n_pairs)


def _pairs(v_i, v_j, curve_j, exclude=None, max_samples: int = 200_000):
    """i 張有效的通道樣本 → (v_i, R_j, j 是否也有效)。排除遮罩內的像素完全不計。"""
    m = valid_mask(v_i)
    if exclude is not None:
        m &= ~exclude
    mj = valid_mask(v_j)[m]
    a, b = v_i[m], v_j[m]
    ok = (mj[:, None] & (b > LO) & (b < HI)).ravel()
    a, b = a.ravel(), b.ravel()
    ch = (a > LO) & (a < HI)
    a, b, ok = a[ch], b[ch], ok[ch]
    step = max(1, a.size // max_samples)
    return a[::step], curve_j(b[::step]), ok[::step]


def _max_spanning_tree(W: np.ndarray, root: int) -> list[tuple[int, int]]:
    """Prim：回傳 (子, 父) 依加入順序。"""
    n = len(W)
    inside, edges = {root}, []
    while len(inside) < n:
        best = max(((W[i, j], i, j) for i in range(n) if i not in inside for j in inside))
        _, i, j = best
        inside.add(i); edges.append((i, j))
    return edges


@dataclass
class Calibration:
    curves: list[Curve]
    root: int
    tree: list[tuple[int, int]]
    overlap: np.ndarray
    ev: list[float] = field(default_factory=list)


def calibrate_chain(vs: list[np.ndarray], exclude: dict | None = None,
                    partner_frac: float = 0.6, width: int = 640) -> Calibration:
    """vs：已對齊的編碼影像（內部縮到 width 寬）。exclude[(i,j)] = 布林遮罩（不採用的像素，任意尺寸）。"""
    vs = [_small(v, width) for v in vs]
    if exclude:
        h, w = vs[0].shape[:2]
        exclude = {k: cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                   for k, m in exclude.items()}
    n = len(vs)
    masks = [valid_mask(v) for v in vs]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            W[i, j] = W[j, i] = (masks[i] & masks[j]).mean()
    root = int(np.argmax([m.mean() for m in masks]))
    tree = _max_spanning_tree(W, root)
    curves: list[Curve | None] = [None] * n
    curves[root] = srgb_curve()
    for i, j in tree:
        # 夥伴：樹上的父節點，加上共同有效面積夠大的其他已校準張
        done = [k for k in range(n) if curves[k] is not None]
        partners = [k for k in done if k == j or W[i, k] >= partner_frac * W[i, j]]
        samples = [_pairs(vs[i], vs[k], curves[k], exclude.get((i, k)) if exclude else None)
                   for k in partners]
        c = fit_curve(samples)
        ex = exclude.get((i, j)) if exclude else None
        if not c.ok:  # 退化：只估單一曝光比例
            c = _scale_only(vs[i], vs[j], curves[j], ex)
        curves[i] = c
    cal = Calibration(curves, root, tree, W)
    mid = srgb_to_linear(np.float32(0.5))
    cal.ev = [float(np.log2(mid / max(c(np.float32(0.5)), 1e-9))) for c in curves]
    return cal


def _scale_only(v_i, v_j, curve_j, ex) -> Curve:
    base = srgb_curve()
    m = (luma(v_i) > 0.01) & (luma(v_j) > 0.01) & (maxch(v_i) < 0.98) & (maxch(v_j) < 0.98)
    if ex is not None:
        m &= ~ex
    if m.sum() < 50:
        return Curve(base.lut, ok=False)
    k = np.median(np.log(luma(curve_j(v_j[m])) + LOG_EPS) - np.log(luma(base(v_i[m])) + LOG_EPS))
    return Curve(base.lut * np.float32(np.exp(k)), ok=False, n_pairs=int(m.sum()))


# ---------- 光狀態差異 ----------
def _small(x: np.ndarray, width: int = 512) -> np.ndarray:
    h, w = x.shape[:2]
    if w <= width:
        return x
    return cv2.resize(x, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def light_difference(v_i, v_j, c_i: Curve, c_j: Curve, thr: float = np.log(1.15),
                     width: int = 512, tol: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """回傳 (i 比 j 亮, j 比 i 亮, 可比較) 三張遮罩（width 寬的縮圖）。
    過曝像素視為下限、過暗像素視為上限；只在界限已足以判定時才算差異。"""
    vi, vj = _small(v_i, width), _small(v_j, width)
    Yi = np.log(luma(c_i(vi)) + LOG_EPS); Yj = np.log(luma(c_j(vj)) + LOG_EPS)
    Yi = cv2.GaussianBlur(Yi, (0, 0), 1.5); Yj = cv2.GaussianBlur(Yj, (0, 0), 1.5)
    ok_i, ok_j = valid_mask(vi), valid_mask(vj)
    cl_i, cl_j = clip_mask(vi), clip_mask(vj)
    dk_i, dk_j = dark_mask(vi), dark_mask(vj)
    up_i = np.log(luma(c_i(np.full(3, LO, np.float32))) + LOG_EPS)  # 過暗像素的上限
    up_j = np.log(luma(c_j(np.full(3, LO, np.float32))) + LOG_EPS)
    # 容忍殘餘對位誤差：須比對方鄰域內最亮值還亮才算（燈光外溢平滑，不受影響；邊緣錯位會被吸收）
    ker_t = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    Yi_max, Yj_max = cv2.dilate(Yi, ker_t), cv2.dilate(Yj, ker_t)
    both = ok_i & ok_j
    i_br = ((both | (cl_i & ok_j)) & (Yi - Yj_max > thr)) | (ok_i & dk_j & (Yi - up_j > thr))
    j_br = ((both | (cl_j & ok_i)) & (Yj - Yi_max > thr)) | (ok_j & dk_i & (Yj - up_i > thr))
    ker = np.ones((3, 3), np.uint8)
    i_br = cv2.morphologyEx(i_br.astype(np.uint8), cv2.MORPH_OPEN, ker).astype(bool)
    j_br = cv2.morphologyEx(j_br.astype(np.uint8), cv2.MORPH_OPEN, ker).astype(bool)
    comp = ok_i | ok_j
    return i_br, j_br, comp


def difference_matrix(vs, curves, width: int = 512, thr: float = np.log(1.15)):
    n = len(vs)
    D = np.zeros((n, n)); B = np.zeros((n, n))
    masks = {}
    for i in range(n):
        for j in range(i + 1, n):
            ib, jb, comp = light_difference(vs[i], vs[j], curves[i], curves[j], thr, width)
            c = max(comp.sum(), 1)
            B[i, j], B[j, i] = ib.sum() / c, jb.sum() / c
            D[i, j] = D[j, i] = (ib | jb).sum() / c
            masks[(i, j)] = masks[(j, i)] = ib | jb
    return D, B, masks


def dilate_masks(masks: dict, size: int = 5) -> dict:
    """燈光差異遮罩外擴（涵蓋外溢的過渡區），供第二輪校準排除。"""
    ker = np.ones((size, size), np.uint8)
    return {k: cv2.dilate(m.astype(np.uint8), ker).astype(bool) for k, m in masks.items()}


# ---------- 光狀態分群 ----------
@dataclass
class States:
    members: list[list[int]]
    ambient: int
    kinds: list[str]          # "ambient" / "light" / "variant"


def cluster_states(D: np.ndarray, B: np.ndarray, thr: float = 0.015,
                   light_ratio: float = 3.0) -> States:
    n = len(D)
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        best, pair = np.inf, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = max(D[i, j] for i in clusters[a] for j in clusters[b])
                if d < best:
                    best, pair = d, (a, b)
        if best >= thr:
            break
        a, b = pair
        clusters[a] = sorted(clusters[a] + clusters[b]); del clusters[b]
    clusters.sort(key=lambda c: c[0])

    def mean_b(sa, sb):
        return float(np.mean([B[i, j] for i in sa for j in sb]))

    k = len(clusters)
    score = [sum(mean_b(clusters[s], clusters[o]) - mean_b(clusters[o], clusters[s])
                 for o in range(k) if o != s) for s in range(k)]
    # 平手時選張數多者（環境光通常拍最多段）
    amb = min(range(k), key=lambda s: (round(score[s], 4), -len(clusters[s])))
    kinds = []
    for s in range(k):
        if s == amb:
            kinds.append("ambient")
        else:
            up, down = mean_b(clusters[s], clusters[amb]), mean_b(clusters[amb], clusters[s])
            kinds.append("light" if up > light_ratio * down else "variant")
    return States(clusters, amb, kinds)


# ---------- 合成 ----------
def hdr_weight(v: np.ndarray) -> np.ndarray:
    """hat 權重：中間調高，接近過曝或過暗趨近 0。"""
    m = maxch(v); y = luma(v)
    w_hi = np.clip((HI - m) / 0.20, 0, 1)
    w_lo = np.clip((y - LO * 0.5) / 0.20, 0, 1)
    hat = 1 - np.abs(2 * y - 1) ** 2
    return (w_hi * w_lo * np.maximum(hat, 0.05)).astype(np.float32)


def hdr_merge(frames, curves: list[Curve], evs: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """frames：可迭代 (index, 已對齊編碼影像)。回傳 (輻射量 RGB, 權重和)。
    全部過曝 → 取最低曝光那張；全部過暗 → 取最高曝光那張。"""
    num = den = None
    lo_img = hi_img = None
    lo_ev, hi_ev = np.inf, -np.inf
    for i, v in frames:
        R = curves[i](v)
        w = hdr_weight(v)
        if num is None:
            num = np.zeros_like(R); den = np.zeros(R.shape[:2], np.float32)
        num += R * w[..., None]; den += w
        if evs[i] < lo_ev:
            lo_ev, lo_img = evs[i], (R, v)
        if evs[i] > hi_ev:
            hi_ev, hi_img = evs[i], (R, v)
    out = num / np.maximum(den, 1e-6)[..., None]
    weak = den < 1e-3
    if weak.any():
        bright = clip_mask(hi_img[1]) if hi_img is not None else np.zeros_like(weak)
        fb = np.where(bright[..., None], lo_img[0], hi_img[0])
        out[weak] = fb[weak]
    return out.astype(np.float32), den


def light_layer(R_state: np.ndarray, R_amb: np.ndarray, den_state: np.ndarray,
                floor: float = 0.05) -> np.ndarray:
    """燈光層 Δ ≥ 0。低於環境光 floor 比例的差視為雜訊；狀態權重過低處不採用。"""
    d = R_state - R_amb * (1 + floor)
    conf = np.clip(den_state / 0.05, 0, 1)[..., None]
    return (np.maximum(d, 0) * conf).astype(np.float32)
