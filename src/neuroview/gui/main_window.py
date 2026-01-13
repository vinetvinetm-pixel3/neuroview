"""
MainWindow (CustomTkinter)

- Creates the layout (sidebar + video preview + status)
- Calls SessionController for load/start/pause/stop
- Schedules frame processing with .after() (no threads yet, same strategy as legacy)
"""

from __future__ import annotations

import time
import customtkinter as ctk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image

from neuroview.core.session_controller import SessionController


class MainWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("NeuroView")
        self.geometry("1280x720")
        self.minsize(1100, 650)

        self.controller = SessionController()

        self._build_layout()

        # Schedule UI refresh for metrics
        self.after(200, self._refresh_metrics)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)  # sidebar
        self.grid_columnconfigure(1, weight=1)  # content
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, corner_radius=16, width=300)
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=12, pady=12)
        self.sidebar.grid_propagate(False)

        title = ctk.CTkLabel(self.sidebar, text="NeuroView", font=ctk.CTkFont(size=22, weight="bold"))
        title.pack(padx=16, pady=(16, 4), anchor="w")

        sub = ctk.CTkLabel(self.sidebar, text="Prototype → Modular App", text_color=("gray70", "gray60"))
        sub.pack(padx=16, pady=(0, 12), anchor="w")

        ctk.CTkButton(self.sidebar, text="Load Video", command=self._on_load_video).pack(padx=16, pady=6, fill="x")
        ctk.CTkButton(self.sidebar, text="Use Camera", command=self._on_use_camera).pack(padx=16, pady=6, fill="x")
        ctk.CTkButton(self.sidebar, text="Start", command=self._on_start).pack(padx=16, pady=6, fill="x")
        ctk.CTkButton(self.sidebar, text="Pause/Resume", command=self._on_pause).pack(padx=16, pady=6, fill="x")
        ctk.CTkButton(self.sidebar, text="Stop", command=self._on_stop).pack(padx=16, pady=6, fill="x")

        # Metrics box
        metrics = ctk.CTkFrame(self.sidebar, corner_radius=16)
        metrics.pack(padx=16, pady=(14, 10), fill="x")

        ctk.CTkLabel(metrics, text="Live Metrics", font=ctk.CTkFont(size=14, weight="bold")).pack(
            padx=12, pady=(10, 6), anchor="w"
        )

        self.m_time = ctk.CTkLabel(metrics, text="Time: 0.00 s")
        self.m_time.pack(padx=12, pady=2, anchor="w")

        self.m_dist = ctk.CTkLabel(metrics, text="Distance: 0.00 px")
        self.m_dist.pack(padx=12, pady=2, anchor="w")

        self.m_speed = ctk.CTkLabel(metrics, text="Speed: 0.00 px/s")
        self.m_speed.pack(padx=12, pady=2, anchor="w")

        self.m_zone = ctk.CTkLabel(metrics, text="Zone: -")
        self.m_zone.pack(padx=12, pady=(2, 10), anchor="w")

        # Content: video preview
        self.content = ctk.CTkFrame(self, corner_radius=16)
        self.content.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.content, text="Load a video to start preview.")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        # Status bar
        self.status_var = ctk.StringVar(value="Ready.")
        status = ctk.CTkLabel(self, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))

    # ----------------------------
    # Button handlers
    # ----------------------------
    def _on_load_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.controller.open_video(path)
            self.status_var.set(f"Video loaded: {path}")
            messagebox.showinfo("Loaded", path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_use_camera(self) -> None:
        try:
            self.controller.open_camera(0)
            self.status_var.set("Camera opened.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_start(self) -> None:
        try:
            # ROI selection will open OpenCV window (legacy behavior)
            self.controller.start()
            self.status_var.set("Running...")
            self._schedule_next_frame()
        except Exception as e:
            messagebox.showwarning("Start cancelled", str(e))

    def _on_pause(self) -> None:
        self.controller.toggle_pause()
        self.status_var.set("Paused." if self.controller.paused else "Running...")

        # If resuming, re-schedule frames
        if not self.controller.paused and self.controller.running:
            self._schedule_next_frame()

    def _on_stop(self) -> None:
        self.controller.stop()
        self.status_var.set("Stopped.")

    # ----------------------------
    # Main loop (after-based, like legacy)
    # ----------------------------
    def _schedule_next_frame(self) -> None:
        if not self.controller.running:
            return
        if self.controller.paused:
            self.after(50, self._schedule_next_frame)
            return

        # Process one frame
        frame_bgr = self.controller.process_next_frame()
        if frame_bgr is not None:
            self._render_frame(frame_bgr)

        # Delay similar to your logic (video fps vs camera)
        if self.controller.source == "video":
            delay_ms = max(int(self.controller.frame_interval_s * 1000), 1)
        else:
            delay_ms = 20

        self.after(delay_ms, self._schedule_next_frame)

    def _render_frame(self, frame_bgr) -> None:
        # Convert BGR -> RGB -> PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Resize to current label size (keeps UI fluid)
        w = max(self.video_label.winfo_width(), 200)
        h = max(self.video_label.winfo_height(), 200)
        img = img.resize((w, h))

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img  # prevent GC

    def _refresh_metrics(self) -> None:
        m = self.controller.metrics
        self.m_time.configure(text=f"Time: {m.run_time_s:.2f} s")
        self.m_dist.configure(text=f"Distance: {m.total_distance_px:.2f} px")
        self.m_speed.configure(text=f"Speed: {m.inst_speed_px_s:.2f} px/s")
        self.m_zone.configure(text=f"Zone: {m.current_zone}")

        self.after(200, self._refresh_metrics)
