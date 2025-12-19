import tkinter as tk
from tkinter import filedialog, ttk
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from config import DetectionParams
import core
import utils


def bgr_to_tk_image(bgr: np.ndarray, max_size: Tuple[int, int]) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img.thumbnail(max_size, utils.LANCZOS)
    return ImageTk.PhotoImage(img)


def gray_to_tk_image(gray_u8: np.ndarray, max_size: Tuple[int, int]) -> ImageTk.PhotoImage:
    img = Image.fromarray(gray_u8, mode="L")
    img.thumbnail(max_size, utils.LANCZOS)
    return ImageTk.PhotoImage(img)


class FireMonitoringApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("火灾检测（RGB-HSI + 形态学 + 面积阈值）")
        self.root.geometry("1200x780")

        self._cap: Optional[cv2.VideoCapture] = None
        self._video_after_id: Optional[str] = None
        self._current_bgr: Optional[np.ndarray] = None

        self.params = DetectionParams()

        self._build_ui()
        self._update_param_labels()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(outer)
        top.pack(fill=tk.X)

        ttk.Button(top, text="打开图片", command=self.open_image).pack(side=tk.LEFT)
        ttk.Button(top, text="打开视频", command=self.open_video).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="停止视频", command=self.stop_video).pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="等待输入…")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=(16, 0))

        mid = ttk.Frame(outer)
        mid.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        left = ttk.Frame(mid)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        params_box = ttk.LabelFrame(left, text="参数", padding=10)
        params_box.pack(fill=tk.X)

        self.fire_ratio_var = tk.DoubleVar(value=self.params.fire_candidate_ratio_threshold)
        self.fire_area_var = tk.DoubleVar(value=self.params.fire_area_ratio_threshold)
        self.smoke_dist_var = tk.DoubleVar(value=self.params.smoke_distance_threshold)
        self.smoke_area_var = tk.DoubleVar(value=self.params.smoke_area_ratio_threshold)
        self.kernel_var = tk.IntVar(value=self.params.morph_kernel_size)

        self.fire_ratio_lbl = ttk.Label(params_box, text="")
        self.fire_area_lbl = ttk.Label(params_box, text="")
        self.smoke_dist_lbl = ttk.Label(params_box, text="")
        self.smoke_area_lbl = ttk.Label(params_box, text="")
        self.kernel_lbl = ttk.Label(params_box, text="")

        self.fire_ratio_lbl.pack(anchor=tk.W)
        ttk.Scale(
            params_box,
            from_=0.01,
            to=0.90,
            orient=tk.HORIZONTAL,
            variable=self.fire_ratio_var,
            command=lambda _=None: self._on_params_change(),
        ).pack(fill=tk.X)

        self.fire_area_lbl.pack(anchor=tk.W, pady=(8, 0))
        ttk.Scale(
            params_box,
            from_=0.0005,
            to=0.05,
            orient=tk.HORIZONTAL,
            variable=self.fire_area_var,
            command=lambda _=None: self._on_params_change(),
        ).pack(fill=tk.X)

        self.smoke_dist_lbl.pack(anchor=tk.W, pady=(8, 0))
        ttk.Scale(
            params_box,
            from_=0.05,
            to=0.60,
            orient=tk.HORIZONTAL,
            variable=self.smoke_dist_var,
            command=lambda _=None: self._on_params_change(),
        ).pack(fill=tk.X)

        self.smoke_area_lbl.pack(anchor=tk.W, pady=(8, 0))
        ttk.Scale(
            params_box,
            from_=0.0005,
            to=0.10,
            orient=tk.HORIZONTAL,
            variable=self.smoke_area_var,
            command=lambda _=None: self._on_params_change(),
        ).pack(fill=tk.X)

        self.kernel_lbl.pack(anchor=tk.W, pady=(8, 0))
        ttk.Scale(
            params_box,
            from_=3,
            to=15,
            orient=tk.HORIZONTAL,
            variable=self.kernel_var,
            command=lambda _=None: self._on_params_change(),
        ).pack(fill=tk.X)

        actions = ttk.LabelFrame(left, text="输出", padding=10)
        actions.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(actions, text="对当前帧/图像重新检测", command=self.run_detection).pack(fill=tk.X)

        self.result_var = tk.StringVar(value="")
        ttk.Label(actions, textvariable=self.result_var, justify=tk.LEFT).pack(fill=tk.X, pady=(8, 0))

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_result = ttk.Frame(self.notebook)
        self.tab_channels = ttk.Frame(self.notebook)
        self.tab_features = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_result, text="结果")
        self.notebook.add(self.tab_channels, text="RGB分量")
        self.notebook.add(self.tab_features, text="特征图")

        self._img_max = (520, 360)

        self._panel_result_a = ttk.Label(self.tab_result)
        self._panel_result_b = ttk.Label(self.tab_result)
        self._panel_result_a.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self._panel_result_b.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.tab_result.columnconfigure(0, weight=1)
        self.tab_result.columnconfigure(1, weight=1)
        self.tab_result.rowconfigure(0, weight=1)

        self._panel_r = ttk.Label(self.tab_channels)
        self._panel_g = ttk.Label(self.tab_channels)
        self._panel_b = ttk.Label(self.tab_channels)
        self._panel_r.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self._panel_g.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self._panel_b.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        for c in range(3):
            self.tab_channels.columnconfigure(c, weight=1)
        self.tab_channels.rowconfigure(0, weight=1)

        self._panel_fire = ttk.Label(self.tab_features)
        self._panel_smoke = ttk.Label(self.tab_features)
        self._panel_fire.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self._panel_smoke.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.tab_features.columnconfigure(0, weight=1)
        self.tab_features.columnconfigure(1, weight=1)
        self.tab_features.rowconfigure(0, weight=1)

        self._tk_refs: List[ImageTk.PhotoImage] = []

    def _on_params_change(self) -> None:
        k = int(round(self.kernel_var.get()))
        if k % 2 == 0:
            k += 1
        k = int(np.clip(k, 3, 31))

        self.params = DetectionParams(
            fire_candidate_ratio_threshold=float(self.fire_ratio_var.get()),
            fire_area_ratio_threshold=float(self.fire_area_var.get()),
            smoke_distance_threshold=float(self.smoke_dist_var.get()),
            smoke_area_ratio_threshold=float(self.smoke_area_var.get()),
            morph_kernel_size=k,
        )
        self.kernel_var.set(k)
        self._update_param_labels()

    def _update_param_labels(self) -> None:
        self.fire_ratio_lbl.configure(text=f"火焰候选比例阈值: {self.params.fire_candidate_ratio_threshold:.2f}")
        self.fire_area_lbl.configure(text=f"火焰面积阈值(占比): {self.params.fire_area_ratio_threshold:.4f}")
        self.smoke_dist_lbl.configure(text=f"烟雾相似度阈值D: {self.params.smoke_distance_threshold:.2f}")
        self.smoke_area_lbl.configure(text=f"烟雾面积阈值(占比): {self.params.smoke_area_ratio_threshold:.4f}")
        self.kernel_lbl.configure(text=f"形态学核大小: {self.params.morph_kernel_size}")

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("Image", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            self.status_var.set("图片读取失败")
            return

        self.stop_video()
        self._current_bgr = bgr
        self.status_var.set(f"已加载图片: {path}")
        self.run_detection()

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[
                ("Video", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.stop_video()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.status_var.set("视频打开失败")
            return

        self._cap = cap
        self.status_var.set(f"视频播放中: {path}")
        self._video_loop()

    def stop_video(self) -> None:
        if self._video_after_id is not None:
            try:
                self.root.after_cancel(self._video_after_id)
            except Exception:
                pass
            self._video_after_id = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _video_loop(self) -> None:
        if self._cap is None:
            return

        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.status_var.set("视频结束")
            self.stop_video()
            return

        self._current_bgr = frame
        self.run_detection()
        self._video_after_id = self.root.after(30, self._video_loop)

    def run_detection(self) -> None:
        if self._current_bgr is None:
            self.status_var.set("请先打开图片或视频")
            return

        bgr = self._current_bgr
        flame = core.detect_flame(bgr, self.params)
        smoke = core.detect_smoke(bgr, self.params)

        overlay = core.overlay_mask(bgr, flame["mask"], smoke["mask"])
        fire_detected = bool(flame["detected"])
        smoke_detected = bool(smoke["detected"])
        fire = fire_detected or smoke_detected

        lines: List[str] = []
        lines.append(f"火焰: {'是' if fire_detected else '否'}")
        lines.append(
            f"  候选比例={flame.get('candidate_ratio', 0.0):.4f}  最大连通域={flame.get('max_area', 0)}"
        )
        lines.append(f"烟雾: {'是' if smoke_detected else '否'}")
        lines.append(
            f"  基准点数={smoke.get('ref_count', 0)}  最大连通域={smoke.get('max_area', 0)}"
        )
        lines.append(f"最终判定: {'火灾' if fire else '正常'}")
        self.result_var.set("\n".join(lines))

        self._tk_refs.clear()

        img_a = bgr_to_tk_image(bgr, self._img_max)
        img_b = bgr_to_tk_image(overlay, self._img_max)
        self._panel_result_a.configure(image=img_a, text="原始图", compound=tk.TOP)
        self._panel_result_b.configure(image=img_b, text="检测叠加图", compound=tk.TOP)
        self._tk_refs.extend([img_a, img_b])

        r = bgr[:, :, 2]
        g = bgr[:, :, 1]
        b = bgr[:, :, 0]
        r_img = gray_to_tk_image(r, self._img_max)
        g_img = gray_to_tk_image(g, self._img_max)
        b_img = gray_to_tk_image(b, self._img_max)
        self._panel_r.configure(image=r_img, text="R", compound=tk.TOP)
        self._panel_g.configure(image=g_img, text="G", compound=tk.TOP)
        self._panel_b.configure(image=b_img, text="B", compound=tk.TOP)
        self._tk_refs.extend([r_img, g_img, b_img])

        fire_img = gray_to_tk_image(flame["mask"], self._img_max)
        smoke_img = gray_to_tk_image(smoke["mask"], self._img_max)
        self._panel_fire.configure(image=fire_img, text="火焰候选/区域", compound=tk.TOP)
        self._panel_smoke.configure(image=smoke_img, text="烟雾候选/区域", compound=tk.TOP)
        self._tk_refs.extend([fire_img, smoke_img])
