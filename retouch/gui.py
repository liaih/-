"""快速修圖 — 桌面介面（PySide6）。
匯入照片（資料夾或多選檔案，可拖放）→ 自動修圖（分群、HDR、自動曝光、套風格）→ 預覽原圖／成品 → 批次匯出。
燈光合成（各燈單獨開關）尚未以實拍驗證，介面標示「待開發」。"""
from __future__ import annotations
import sys
import traceback
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSize, QUrl
from PySide6.QtGui import QImage, QPixmap, QIcon, QDesktopServices
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QListWidget, QListWidgetItem, QFileDialog, QProgressBar, QPlainTextEdit,
                               QComboBox, QSplitter, QMessageBox, QGroupBox, QCheckBox, QLineEdit)
from .imageio import IMG_EXT, load_image, resize_max
from .pipeline import run, PRESETS

APP_NAME = "快速修圖"


def to_pixmap(rgb: np.ndarray, max_side: int) -> QPixmap:
    rgb = resize_max(rgb, max_side)
    a = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    h, w = a.shape[:2]
    return QPixmap.fromImage(QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy())


class Worker(QThread):
    log = Signal(str)
    progress = Signal(int, int)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, paths, out_dir, tiff):
        super().__init__()
        self.paths, self.out_dir, self.tiff = paths, out_dir, tiff

    def run(self):
        try:
            rep = run(self.paths, self.out_dir, tiff=self.tiff, previews=False,
                      log=lambda *a: self.log.emit(" ".join(str(x) for x in a)),
                      progress=lambda i, n: self.progress.emit(i, n))
            self.done.emit(rep)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ImageView(QLabel):
    """等比縮放顯示的圖片框。"""

    def __init__(self, title):
        super().__init__()
        self.title, self._rgb = title, None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 220)
        self.setStyleSheet("background:#1e1e1e;color:#aaa;border-radius:6px;")
        self.setText(title)

    def set_image(self, rgb):
        self._rgb = rgb
        self._render()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    def _render(self):
        if self._rgb is None:
            return
        pm = to_pixmap(self._rgb, 1600)
        self.setPixmap(pm.scaled(self.size() - QSize(8, 8), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — 室內攝影批次修圖")
        self.resize(1400, 860)
        self.setAcceptDrops(True)
        self.paths: list[Path] = []
        self.report, self.worker = None, None

        # 左：匯入
        left = QVBoxLayout()
        b_dir, b_files = QPushButton("匯入資料夾…"), QPushButton("加入照片…")
        b_clear = QPushButton("清除")
        b_dir.clicked.connect(self.pick_dir); b_files.clicked.connect(self.pick_files); b_clear.clicked.connect(self.clear)
        row = QHBoxLayout(); row.addWidget(b_dir); row.addWidget(b_files); row.addWidget(b_clear)
        left.addLayout(row)
        self.lst_in = QListWidget(); self.lst_in.setIconSize(QSize(96, 64))
        left.addWidget(QLabel("匯入的照片（可直接拖放檔案或資料夾進視窗）"))
        left.addWidget(self.lst_in, 1)

        box = QGroupBox("輸出設定")
        bl = QVBoxLayout(box)
        r1 = QHBoxLayout()
        self.ed_out = QLineEdit(); self.ed_out.setPlaceholderText("輸出資料夾（預設：照片資料夾\\修圖輸出）")
        b_out = QPushButton("選擇…"); b_out.clicked.connect(self.pick_out)
        r1.addWidget(self.ed_out, 1); r1.addWidget(b_out)
        bl.addLayout(r1)
        r2 = QHBoxLayout()
        self.cmb_fmt = QComboBox(); self.cmb_fmt.addItems(["JPG（sRGB、品質 95）", "TIFF 16-bit"])
        r2.addWidget(QLabel("格式")); r2.addWidget(self.cmb_fmt, 1)
        bl.addLayout(r2)
        style = PRESETS / "style_default.xmp"
        bl.addWidget(QLabel(f"風格：{style.name}（LR 預設集，已依成品校準）"))
        lights = QCheckBox("燈光合成（吊燈／軌道燈／層板燈）— 待開發")
        lights.setEnabled(False)
        lights.setToolTip("需同機位開關燈實拍驗證後開放")
        bl.addWidget(lights)
        left.addWidget(box)

        self.btn_run = QPushButton("開始自動修圖")
        self.btn_run.setMinimumHeight(44)
        self.btn_run.setStyleSheet("font-size:16px;font-weight:bold;")
        self.btn_run.clicked.connect(self.start)
        left.addWidget(self.btn_run)
        self.bar = QProgressBar(); left.addWidget(self.bar)
        self.txt_log = QPlainTextEdit(); self.txt_log.setReadOnly(True); self.txt_log.setMaximumHeight(140)
        left.addWidget(self.txt_log)
        wl = QWidget(); wl.setLayout(left)

        # 右：結果與預覽
        right = QVBoxLayout()
        right.addWidget(QLabel("修圖結果（每個機位一張）"))
        self.lst_out = QListWidget(); self.lst_out.setViewMode(QListWidget.IconMode)
        self.lst_out.setIconSize(QSize(180, 120)); self.lst_out.setMaximumHeight(190)
        self.lst_out.setFlow(QListWidget.LeftToRight); self.lst_out.setWrapping(False)
        self.lst_out.currentRowChanged.connect(self.show_result)
        right.addWidget(self.lst_out)
        views = QHBoxLayout()
        self.v_before, self.v_after = ImageView("原圖"), ImageView("成品")
        self.cap_before, self.cap_after = QLabel("原圖"), QLabel("成品")
        for cap, v in ((self.cap_before, self.v_before), (self.cap_after, self.v_after)):
            col = QVBoxLayout(); cap.setStyleSheet("font-weight:bold;"); col.addWidget(cap); col.addWidget(v, 1)
            views.addLayout(col, 1)
        right.addLayout(views, 1)
        self.lbl_info = QLabel(""); self.lbl_info.setWordWrap(True)
        right.addWidget(self.lbl_info)
        self.btn_open = QPushButton("開啟輸出資料夾"); self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.out_dir()))))
        right.addWidget(self.btn_open)
        wr = QWidget(); wr.setLayout(right)

        sp = QSplitter(); sp.addWidget(wl); sp.addWidget(wr); sp.setSizes([420, 980])
        self.setCentralWidget(sp)

    # ---------- 匯入 ----------
    def add_paths(self, items):
        new = []
        for p in map(Path, items):
            if p.is_dir():
                new += [q for q in sorted(p.iterdir()) if q.suffix.lower() in IMG_EXT]
            elif p.suffix.lower() in IMG_EXT:
                new.append(p)
        for p in new:
            if p not in self.paths:
                self.paths.append(p)
                it = QListWidgetItem(QIcon(to_pixmap(load_image(p, 192), 192)), p.name)
                self.lst_in.addItem(it)
            QApplication.processEvents()
        self.statusBar().showMessage(f"已匯入 {len(self.paths)} 張")

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "選擇照片資料夾")
        if d:
            self.add_paths([d])

    def pick_files(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "選擇照片", "", "影像 (*.jpg *.jpeg *.tif *.tiff *.png *.cr3 *.cr2 *.dng)")
        self.add_paths(fs)

    def pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾")
        if d:
            self.ed_out.setText(d)

    def clear(self):
        self.paths.clear(); self.lst_in.clear()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls()])

    def out_dir(self) -> Path:
        t = self.ed_out.text().strip()
        return Path(t) if t else self.paths[0].parent / "修圖輸出"

    # ---------- 執行 ----------
    def start(self):
        if not self.paths:
            QMessageBox.information(self, APP_NAME, "請先匯入照片。")
            return
        self.btn_run.setEnabled(False); self.btn_open.setEnabled(False)
        self.lst_out.clear(); self.txt_log.clear(); self.bar.setRange(0, 0)
        self.worker = Worker(list(self.paths), self.out_dir(), self.cmb_fmt.currentIndex() == 1)
        self.worker.log.connect(self.txt_log.appendPlainText)
        self.worker.progress.connect(lambda i, n: (self.bar.setRange(0, n), self.bar.setValue(i)))
        self.worker.done.connect(self.finished)
        self.worker.failed.connect(self.failed)
        self.worker.start()

    def failed(self, msg):
        self.btn_run.setEnabled(True); self.bar.setRange(0, 1); self.bar.setValue(0)
        self.txt_log.appendPlainText(msg)
        QMessageBox.critical(self, APP_NAME, "修圖失敗，詳見下方紀錄。")

    def finished(self, rep):
        self.report = rep
        self.btn_run.setEnabled(True); self.btn_open.setEnabled(True)
        out = self.out_dir()
        for g in rep["groups"]:
            fn = out / g["outputs"][0]["file"]
            it = QListWidgetItem(QIcon(to_pixmap(load_image(fn, 360), 360)), f"{g['name']}（{len(g['files'])} 張）")
            self.lst_out.addItem(it)
        self.txt_log.appendPlainText(f"完成：{len(rep['groups'])} 個機位，已輸出到 {out}")
        self.statusBar().showMessage(f"完成，輸出到 {out}")
        if rep["groups"]:
            self.lst_out.setCurrentRow(0)

    def show_result(self, row):
        if self.report is None or row < 0:
            return
        g = self.report["groups"][row]
        by_name = {p.name: p for p in self.paths}
        # 原圖：曝光最接近基準（ev≈0）的那張
        ex = min(g["exposure"], key=lambda e: abs(e["ev"]))
        self.v_before.set_image(load_image(by_name[ex["file"]], 1600))
        self.v_after.set_image(load_image(self.out_dir() / g["outputs"][0]["file"], 1600))
        self.cap_before.setText(f"原圖（{ex['file']}，曝光基準張）")
        self.cap_after.setText(f"成品（{g['outputs'][0]['file']}）")
        labels = "、".join(f"{e['file']}（{e['label']}）" for e in g["exposure"])
        self.lbl_info.setText(f"<b>{g['outputs'][0]['file']}</b>　自動曝光 {g['outputs'][0]['auto_ev']:+.2f} EV<br>"
                              f"合成來源：{labels}")


def main(argv=None):
    """一般啟動開啟視窗。自動測試：--demo 照片資料夾 --screenshot 截圖.png（匯入、修圖、截圖後結束）。"""
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    w = MainWindow()
    w.show()
    if "--demo" in argv:
        from PySide6.QtCore import QTimer
        demo = argv[argv.index("--demo") + 1]
        shot = argv[argv.index("--screenshot") + 1] if "--screenshot" in argv else None
        status = {"code": 1}

        def finish(code):
            if shot:
                app.processEvents(); w.grab().save(shot)
            status["code"] = code
            app.quit()

        def go():
            w.add_paths([demo]); w.start()
            w.worker.done.connect(lambda r: QTimer.singleShot(500, lambda: finish(0 if r["groups"] else 1)))
            w.worker.failed.connect(lambda m: (print(m), finish(1)))
        QTimer.singleShot(300, go)
        app.exec()
        return status["code"]
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
