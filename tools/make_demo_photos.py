"""產生 CI 用的合成測試照片（同一場景 3 段曝光＋另一場景 2 段），只用來確認 EXE 能啟動並完成流程。"""
import sys
from pathlib import Path
import numpy as np
import cv2

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
for scene in range(2):
    h, w = 800, 1200
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    R = 0.15 + 0.25 * (xx / w) + 0.1 * (yy / h)
    for k in range(25):
        x0, y0 = rng.integers(0, w - 200), rng.integers(0, h - 150)
        R[y0:y0 + rng.integers(40, 150), x0:x0 + rng.integers(40, 200)] = rng.uniform(0.05, 0.9)
    R = cv2.GaussianBlur(R, (0, 0), 1.2)
    R[100:300, 800:1100] = 6.0  # 窗外
    rgb = np.dstack([R * 1.05, R, R * 0.9])
    for e in ((1, 0, -2) if scene == 0 else (0, -2)):
        v = np.clip(rgb * 2.0 ** e, 0, 1) ** (1 / 2.2)
        cv2.imwrite(str(out / f"scene{scene}_{e:+d}.jpg"), (v[..., ::-1] * 255).astype(np.uint8))
print("ok", sorted(p.name for p in out.iterdir()))
