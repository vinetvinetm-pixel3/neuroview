"""
MainWindow (CustomTkinter) - NeuroView MVP (Vector Pose)

UI goals:
- Any-maze-like aesthetics: show ONLY pose vectors (nose-body-tail) with thin lines + 3 points.
- Never draw segmentation contours.
- If tracking confidence is low or tracking is missing, fade out / hide markers
  (do NOT draw in wrong places).
- Inline ROI selection:
    1) Full Area
    2) Select Area (drag)
    3) Reset ROI
- Controls:
    - Play/Start
    - Pause
    - Stop
    - Reset Video
- Exports:
    - CSV
    - PDF
(using native OS dialogs via tkinter.filedialog)
"""

from __future__ import annotations

import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from neuroview.core.session_controller import SessionController


class MainWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("NeuroView")
        self.geometry("1280x720")
        self.minsize(1100, 650)

        self.controller = SessionController()
        self.source_loaded: bool = False

        # Camera preview loop flags
        self._camera_preview_active: bool = False

        # ROI state (GUI-only)
        self._roi_mode: str = "NONE"  # NONE | SELECTING | DEFINED
        self._roi_start_canvas: tuple[int, int] | None = None
        self._roi_end_canvas: tuple[int, int] | None = None

        # Canvas item ids
        self._canvas_image_id: int | None = None
        self._canvas_text_id: int | None = None
        self._roi_rect_id: int | None = None

        self._tk_img: ImageTk.PhotoImage | None = None

        # Render mapping (canvas -> frame)
        self._render_scale: float = 1.0
        self._render_off_x: int = 0
        self._render_off_y: int = 0
        self._render_src_w: int = 1
        self._render_src_h: int = 1
        self._render_disp_w: int = 1
        self._render_disp_h: int = 1

        self._build_layout()
        self._set_controls_state()

        self.after(200, self._refresh_metrics)

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, corner_radius=16, width=320)
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=12, pady=12)
        self.sidebar.grid_propagate(False)

        title = ctk.CTkLabel(self.sidebar, text="NeuroView", font=ctk.CTkFont(size=22, weight="bold"))
        title.pack(padx=16, pady=(16, 4), anchor="w")

        sub = ctk.CTkLabel(self.sidebar, text="Prototype → Modular App", text_color=("gray70", "gray60"))
        sub.pack(padx=16, pady=(0, 12), anchor="w")

        # Source
        self.btn_load_video = ctk.CTkButton(self.sidebar, text="Load Video", command=self._on_load_video)
        self.btn_load_video.pack(padx=16, pady=6, fill="x")

        self.btn_use_camera = ctk.CTkButton(self.sidebar, text="Use Camera", command=self._on_use_camera)
        self.btn_use_camera.pack(padx=16, pady=6, fill="x")

        # ROI box
        roi_box = ctk.CTkFrame(self.sidebar, corner_radius=14)
        roi_box.pack(padx=16, pady=(14, 8), fill="x")

        ctk.CTkLabel(roi_box, text="Arena (ROI)", font=ctk.CTkFont(size=13, weight="bold")).pack(
            padx=12, pady=(10, 8), anchor="w"
        )

        self.btn_roi_full = ctk.CTkButton(roi_box, text="Full Area", command=self._roi_set_full_arena)
        self.btn_roi_full.pack(padx=12, pady=6, fill="x")

        self.btn_roi_select = ctk.CTkButton(roi_box, text="Select Area", command=self._roi_enter_select_mode)
        self.btn_roi_select.pack(padx=12, pady=6, fill="x")

        self.btn_roi_reset = ctk.CTkButton(roi_box, text="Reset ROI", command=self._roi_reset)
        self.btn_roi_reset.pack(padx=12, pady=(6, 12), fill="x")

        # Controls
        self.btn_start = ctk.CTkButton(self.sidebar, text="Play/Start", command=self._on_start, text_color_disabled=("gray80", "gray55"))
        self.btn_start.pack(padx=16, pady=(8, 6), fill="x")

        self.btn_pause = ctk.CTkButton(self.sidebar, text="Pause", command=self._on_pause, text_color_disabled=("gray80", "gray55"))
        self.btn_pause.pack(padx=16, pady=6, fill="x")

        self.btn_stop = ctk.CTkButton(self.sidebar, text="Stop", command=self._on_stop, text_color_disabled=("gray80", "gray55"))
        self.btn_stop.pack(padx=16, pady=6, fill="x")

        self.btn_reset_video = ctk.CTkButton(self.sidebar, text="Reset Video", command=self._on_reset_video, text_color_disabled=("gray80", "gray55"))
        self.btn_reset_video.pack(padx=16, pady=(6, 6), fill="x")

        # Export box
        export_box = ctk.CTkFrame(self.sidebar, corner_radius=14)
        export_box.pack(padx=16, pady=(14, 8), fill="x")

        ctk.CTkLabel(export_box, text="Export", font=ctk.CTkFont(size=13, weight="bold")).pack(
            padx=12, pady=(10, 8), anchor="w"
        )

        self.btn_export_csv = ctk.CTkButton(export_box, text="Export CSV", command=self._on_export_csv)
        self.btn_export_csv.pack(padx=12, pady=6, fill="x")

        self.btn_export_pdf = ctk.CTkButton(export_box, text="Export PDF", command=self._on_export_pdf)
        self.btn_export_pdf.pack(padx=12, pady=(6, 12), fill="x")

        # Metrics
        metrics_box = ctk.CTkFrame(self.sidebar, corner_radius=16)
        metrics_box.pack(padx=16, pady=(14, 10), fill="x")

        ctk.CTkLabel(metrics_box, text="Live Metrics", font=ctk.CTkFont(size=14, weight="bold")).pack(
            padx=12, pady=(10, 6), anchor="w"
        )

        self.lbl_time = ctk.CTkLabel(metrics_box, text="Time: 0.00 s")
        self.lbl_time.pack(padx=12, pady=2, anchor="w")

        self.lbl_dist = ctk.CTkLabel(metrics_box, text="Distance: 0.00 px")
        self.lbl_dist.pack(padx=12, pady=2, anchor="w")

        self.lbl_speed = ctk.CTkLabel(metrics_box, text="Speed: 0.00 px/s")
        self.lbl_speed.pack(padx=12, pady=2, anchor="w")

        self.lbl_speed_avg = ctk.CTkLabel(metrics_box, text="Avg Speed: 0.00 px/s")
        self.lbl_speed_avg.pack(padx=12, pady=2, anchor="w")

        self.lbl_zone = ctk.CTkLabel(metrics_box, text="Zone: -")
        self.lbl_zone.pack(padx=12, pady=2, anchor="w")

        self.lbl_freezing = ctk.CTkLabel(metrics_box, text="Freezing: off (0.0 s)")
        self.lbl_freezing.pack(padx=12, pady=2, anchor="w")

        self.lbl_grooming = ctk.CTkLabel(metrics_box, text="Grooming: off (0.0 s)")
        self.lbl_grooming.pack(padx=12, pady=(2, 10), anchor="w")

        # Content area
        self.content = ctk.CTkFrame(self, corner_radius=16)
        self.content.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(self.content, highlightthickness=0, bd=0, background="#121212")
        self.video_canvas.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        self.video_canvas.bind("<Configure>", self._on_canvas_resize)

        self._canvas_text_id = self.video_canvas.create_text(
            16, 16, anchor="nw", fill="#CFCFCF", text="Load a video to start preview."
        )

        # Status bar
        self.status_var = ctk.StringVar(value="Ready.")
        status = ctk.CTkLabel(self, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))

        self._disable_roi_mouse()

    # -------------------------------------------------------------------------
    # Resizing
    # -------------------------------------------------------------------------
    def _on_canvas_resize(self, event: tk.Event) -> None:
        if self.controller.last_frame is not None:
            self._render_frame(self.controller.last_frame)

    # -------------------------------------------------------------------------
    # Controls state
    # -------------------------------------------------------------------------
    def _set_controls_state(self) -> None:
        running = bool(self.controller.running)
        paused = bool(self.controller.paused)
        roi_ready = bool(self.controller.roi_defined and self.controller.roi is not None)

        self.btn_start.configure(state="normal" if (self.source_loaded and roi_ready and not running) else "disabled")
        self.btn_pause.configure(state="normal" if running else "disabled")
        self.btn_stop.configure(state="normal" if (running or paused) else "disabled")

        can_reset_video = self.source_loaded and (self.controller.source == "video") and not running
        self.btn_reset_video.configure(state="normal" if can_reset_video else "disabled")

        roi_buttons_enabled = self.source_loaded and not running
        state_roi = "normal" if roi_buttons_enabled else "disabled"
        self.btn_roi_full.configure(state=state_roi)
        self.btn_roi_select.configure(state=state_roi)
        self.btn_roi_reset.configure(state=state_roi)

        export_ok = len(getattr(self.controller, "tracking_records", [])) > 0
        self.btn_export_csv.configure(state="normal" if export_ok else "disabled")
        self.btn_export_pdf.configure(state="normal" if export_ok else "disabled")

    # -------------------------------------------------------------------------
    # Camera preview loop
    # -------------------------------------------------------------------------
    def _start_camera_preview(self) -> None:
        if self.controller.cap is None:
            return
        self._camera_preview_active = True
        self._schedule_camera_preview_frame()

    def _stop_camera_preview(self) -> None:
        self._camera_preview_active = False

    def _schedule_camera_preview_frame(self) -> None:
        if not self._camera_preview_active:
            return
        if self.controller.running:
            self._camera_preview_active = False
            return
        if self.controller.cap is None:
            self._camera_preview_active = False
            return

        ret, frame = self.controller.cap.read()
        if ret and frame is not None:
            self.controller.last_frame = frame.copy()
            self._render_frame(frame)
        else:
            self._camera_preview_active = False
            self.status_var.set("Camera preview stopped (no frames).")

        self.after(33, self._schedule_camera_preview_frame)

    # -------------------------------------------------------------------------
    # ROI UX
    # -------------------------------------------------------------------------
    def _roi_enter_select_mode(self) -> None:
        if not self.source_loaded:
            self.status_var.set("Load a video or open the camera first.")
            return

        self._roi_mode = "SELECTING"
        self._roi_start_canvas = None
        self._roi_end_canvas = None
        self._clear_roi_overlay()

        self._enable_roi_mouse()
        self.status_var.set("Select ROI: click & drag on the preview panel.")

    def _roi_set_full_arena(self) -> None:
        if not self.source_loaded:
            self.status_var.set("Load a video or open the camera first.")
            return
        if self.controller.last_frame is None:
            self.status_var.set("No frame available yet.")
            return

        h, w = self.controller.last_frame.shape[:2]
        self.controller.set_roi((0, 0, w, h))
        self._roi_mode = "DEFINED"

        self.status_var.set("ROI set to Full Area. Ready to Start.")
        self._update_roi_overlay_from_controller()
        self._disable_roi_mouse()
        self._set_controls_state()

    def _roi_reset(self) -> None:
        self.controller.clear_roi()
        self._roi_mode = "NONE"
        self._roi_start_canvas = None
        self._roi_end_canvas = None
        self._clear_roi_overlay()
        self._disable_roi_mouse()

        # Re-render clean frame so ROI does not remain visible
        if self.controller.last_frame is not None:
            self.after(0, lambda: self._render_frame(self.controller.last_frame))

        self.status_var.set("ROI cleared. Select an arena region.")
        self._set_controls_state()

    def _enable_roi_mouse(self) -> None:
        self.video_canvas.bind("<ButtonPress-1>", self._roi_mouse_down)
        self.video_canvas.bind("<B1-Motion>", self._roi_mouse_move)
        self.video_canvas.bind("<ButtonRelease-1>", self._roi_mouse_up)

    def _disable_roi_mouse(self) -> None:
        self.video_canvas.unbind("<ButtonPress-1>")
        self.video_canvas.unbind("<B1-Motion>")
        self.video_canvas.unbind("<ButtonRelease-1>")

    def _roi_mouse_down(self, event: tk.Event) -> None:
        if self._roi_mode != "SELECTING":
            return
        self._roi_start_canvas = (event.x, event.y)
        self._roi_end_canvas = (event.x, event.y)
        self._draw_roi_overlay()

    def _roi_mouse_move(self, event: tk.Event) -> None:
        if self._roi_mode != "SELECTING" or self._roi_start_canvas is None:
            return
        self._roi_end_canvas = (event.x, event.y)
        self._draw_roi_overlay()

    def _roi_mouse_up(self, event: tk.Event) -> None:
        if self._roi_mode != "SELECTING" or self._roi_start_canvas is None:
            return

        self._roi_end_canvas = (event.x, event.y)

        roi = self._roi_canvas_to_frame_roi(self._roi_start_canvas, self._roi_end_canvas)
        if roi is None:
            self.status_var.set("Invalid ROI. Drag inside the video area.")
            self._clear_roi_overlay()
            return

        self.controller.set_roi(roi)
        self._roi_mode = "DEFINED"
        self._disable_roi_mouse()

        self.status_var.set("ROI selected. Ready to Start.")
        self._update_roi_overlay_from_controller()
        self._set_controls_state()

    def _roi_canvas_to_frame_roi(self, p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int, int, int] | None:
        x1, y1 = p1
        x2, y2 = p2
        cx1, cx2 = sorted([x1, x2])
        cy1, cy2 = sorted([y1, y2])

        img_left = self._render_off_x
        img_top = self._render_off_y
        img_right = self._render_off_x + self._render_disp_w
        img_bottom = self._render_off_y + self._render_disp_h

        cx1 = max(img_left, min(cx1, img_right))
        cx2 = max(img_left, min(cx2, img_right))
        cy1 = max(img_top, min(cy1, img_bottom))
        cy2 = max(img_top, min(cy2, img_bottom))

        if (cx2 - cx1) < 8 or (cy2 - cy1) < 8:
            return None

        scale = max(self._render_scale, 1e-6)
        fx1 = int((cx1 - self._render_off_x) / scale)
        fy1 = int((cy1 - self._render_off_y) / scale)
        fx2 = int((cx2 - self._render_off_x) / scale)
        fy2 = int((cy2 - self._render_off_y) / scale)

        fx1 = max(0, min(fx1, self._render_src_w - 1))
        fy1 = max(0, min(fy1, self._render_src_h - 1))
        fx2 = max(0, min(fx2, self._render_src_w))
        fy2 = max(0, min(fy2, self._render_src_h))

        w = fx2 - fx1
        h = fy2 - fy1
        if w <= 0 or h <= 0:
            return None

        return (fx1, fy1, w, h)

    def _clear_roi_overlay(self) -> None:
        if self._roi_rect_id is not None:
            self.video_canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

    def _draw_roi_overlay(self) -> None:
        self._clear_roi_overlay()
        if self._roi_start_canvas is None or self._roi_end_canvas is None:
            return

        x1, y1 = self._roi_start_canvas
        x2, y2 = self._roi_end_canvas
        cx1, cx2 = sorted([x1, x2])
        cy1, cy2 = sorted([y1, y2])

        img_left = self._render_off_x
        img_top = self._render_off_y
        img_right = self._render_off_x + self._render_disp_w
        img_bottom = self._render_off_y + self._render_disp_h

        cx1 = max(img_left, min(cx1, img_right))
        cx2 = max(img_left, min(cx2, img_right))
        cy1 = max(img_top, min(cy1, img_bottom))
        cy2 = max(img_top, min(cy2, img_bottom))

        self._roi_rect_id = self.video_canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#3B82F6", width=3)

    def _update_roi_overlay_from_controller(self) -> None:
        self._clear_roi_overlay()
        if not (self.controller.roi_defined and self.controller.roi is not None):
            return

        fx, fy, fw, fh = self.controller.roi
        cx1 = int(self._render_off_x + fx * self._render_scale)
        cy1 = int(self._render_off_y + fy * self._render_scale)
        cx2 = int(self._render_off_x + (fx + fw) * self._render_scale)
        cy2 = int(self._render_off_y + (fy + fh) * self._render_scale)

        self._roi_start_canvas = (cx1, cy1)
        self._roi_end_canvas = (cx2, cy2)
        self._draw_roi_overlay()

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    def _on_export_csv(self) -> None:
        try:
            path = filedialog.asksaveasfilename(
                parent=self,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Export CSV",
            )
            if not path:
                return
            self.controller.export_csv(path)
            self.status_var.set("CSV exported.")
        except Exception as e:
            messagebox.showerror("Export error", str(e), parent=self)

    def _on_export_pdf(self) -> None:
        try:
            path = filedialog.asksaveasfilename(
                parent=self,
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")],
                title="Export PDF",
            )
            if not path:
                return
            self.controller.export_pdf(path)
            self.status_var.set("PDF exported.")
        except Exception as e:
            messagebox.showerror("Export error", str(e), parent=self)

    # -------------------------------------------------------------------------
    # Buttons
    # -------------------------------------------------------------------------
    def _on_load_video(self) -> None:
        try:
            path = filedialog.askopenfilename(
                parent=self,
                title="Select a video",
                filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
            )
            if not path:
                self.status_var.set("Load cancelled.")
                return

            self._stop_camera_preview()
            if self.controller.running or self.controller.paused:
                self.controller.stop()

            self.controller.open_video(path)
            self.source_loaded = True

            # reset ROI state
            self.controller.clear_roi()
            self._roi_mode = "NONE"
            self._clear_roi_overlay()
            self._disable_roi_mouse()

            first = self.controller.read_first_frame()
            if first is not None:
                self.after(0, lambda f=first: self._render_frame(f))
                self.status_var.set("Video loaded. Select ROI (Full Area or Select Area).")
            else:
                self.status_var.set("Video loaded, but could not read first frame.")
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            self.status_var.set("Error loading video.")
        finally:
            self._set_controls_state()

    def _on_use_camera(self) -> None:
        try:
            if self.controller.running or self.controller.paused:
                self.controller.stop()

            self.controller.open_camera(0)
            self.source_loaded = True

            self.controller.clear_roi()
            self._roi_mode = "NONE"
            self._clear_roi_overlay()
            self._disable_roi_mouse()

            first = self.controller.read_first_frame()
            if first is not None:
                self.after(0, lambda f=first: self._render_frame(f))

            self.status_var.set("Camera ready. Select ROI (Full Area or Select Area). Live preview enabled.")
            self._start_camera_preview()
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            self.status_var.set("Error opening camera.")
        finally:
            self._set_controls_state()

    def _on_start(self) -> None:
        try:
            self._stop_camera_preview()
            self.controller.start()
            self.status_var.set("Running…")
            self._set_controls_state()
            self._schedule_next_frame()
        except Exception as e:
            messagebox.showwarning("Start cancelled", str(e), parent=self)
            self.status_var.set("Start cancelled.")
            self._set_controls_state()

    def _on_pause(self) -> None:
        if not self.controller.running:
            self.status_var.set("Pause is available only while running.")
            self._set_controls_state()
            return

        self.controller.toggle_pause()
        if self.controller.paused:
            self.status_var.set("Paused.")
        else:
            self.status_var.set("Running…")
            self._schedule_next_frame()
        self._set_controls_state()

    def _on_stop(self) -> None:
        if not self.controller.running and not self.controller.paused:
            self.status_var.set("Nothing to stop.")
            self._set_controls_state()
            return

        self.controller.stop()
        self.status_var.set("Stopped.")

        # Clear ROI overlay burn-in and re-render clean last frame
        self._clear_roi_overlay()
        if self.controller.last_frame is not None:
            self.after(0, lambda: self._render_frame(self.controller.last_frame))

        if self.controller.source == "camera" and self.controller.cap is not None:
            self._start_camera_preview()

        self._set_controls_state()

    def _on_reset_video(self) -> None:
        try:
            if self.controller.running or self.controller.paused:
                self.controller.stop()

            self.controller.reset_video_position()
            if self.controller.last_frame is not None:
                self.after(0, lambda: self._render_frame(self.controller.last_frame))

            self.status_var.set("Video reset to start.")
        except Exception as e:
            messagebox.showerror("Reset error", str(e), parent=self)
        finally:
            self._set_controls_state()

    # -------------------------------------------------------------------------
    # Tracking loop
    # -------------------------------------------------------------------------
    def _schedule_next_frame(self) -> None:
        if not self.controller.running:
            self._set_controls_state()
            return

        if self.controller.paused:
            self.after(50, self._schedule_next_frame)
            return

        frame_bgr = self.controller.process_next_frame()
        if frame_bgr is not None:
            self._render_frame(frame_bgr)
        else:
            if self.controller.source == "video" and not self.controller.running:
                self.status_var.set("Finished (end of video).")
                self._set_controls_state()
                return

        delay_ms = max(int(self.controller.frame_interval_s * 1000), 1) if self.controller.source == "video" else 20
        self.after(delay_ms, self._schedule_next_frame)

    # -------------------------------------------------------------------------
    # Any-maze-like overlay drawing (NO contours)
    # -------------------------------------------------------------------------
    @staticmethod
    def _draw_pose_overlay(frame_bgr: np.ndarray, body: Optional[object]) -> np.ndarray:
        """
        Draw only the pose vectors:
        - Line: nose->center
        - Line: center->tail
        - 3 circles for points
        Uses confidence to fade when uncertain.
        """
        if body is None:
            return frame_bgr

        # Extract points
        nose = tuple(map(int, body.head_xy))
        center = tuple(map(int, body.center_xy))
        tail = tuple(map(int, body.tail_xy))

        conf = float(getattr(body, "confidence", 1.0))
        conf = max(0.0, min(conf, 1.0))

        # Fade rule: if confidence too low, don't draw at all (prevents misleading UI)
        if conf < 0.35:
            return frame_bgr

        # Create overlay for alpha blending
        overlay = frame_bgr.copy()

        # Professional, subtle colors (BGR)
        nose_col = (80, 255, 120)     # neon green-ish
        center_col = (255, 170, 60)   # warm blue-ish? (note BGR order)
        tail_col = (255, 255, 120)    # soft cyan

        line_col = (240, 240, 240)    # light gray

        # Thin lines
        cv2.line(overlay, nose, center, line_col, 1, cv2.LINE_AA)
        cv2.line(overlay, center, tail, line_col, 1, cv2.LINE_AA)

        # Radius scales with image size a bit
        h, w = frame_bgr.shape[:2]
        r = max(3, int(min(h, w) * 0.006))

        # Draw circles with anti-aliasing
        cv2.circle(overlay, nose, r, nose_col, -1, cv2.LINE_AA)
        cv2.circle(overlay, center, r, center_col, -1, cv2.LINE_AA)
        cv2.circle(overlay, tail, r, tail_col, -1, cv2.LINE_AA)

        # Alpha depends on confidence (subtle fade)
        alpha = 0.35 + 0.65 * conf
        cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0, frame_bgr)
        return frame_bgr

    # -------------------------------------------------------------------------
    # Render frame
    # -------------------------------------------------------------------------
    def _render_frame(self, frame_bgr: np.ndarray) -> None:
        # Draw pose overlay (no contours)
        frame_bgr = self._draw_pose_overlay(frame_bgr, self.controller.last_body)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self._render_src_w, self._render_src_h = img.size

        self.update_idletasks()

        canvas_w = max(self.video_canvas.winfo_width(), 200)
        canvas_h = max(self.video_canvas.winfo_height(), 200)

        scale = min(canvas_w / self._render_src_w, canvas_h / self._render_src_h)
        disp_w = max(1, int(self._render_src_w * scale))
        disp_h = max(1, int(self._render_src_h * scale))
        off_x = (canvas_w - disp_w) // 2
        off_y = (canvas_h - disp_h) // 2

        self._render_scale = scale
        self._render_off_x = off_x
        self._render_off_y = off_y
        self._render_disp_w = disp_w
        self._render_disp_h = disp_h

        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img_resized = img.resize((disp_w, disp_h), resample)

        canvas_img = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
        canvas_img.paste(img_resized, (off_x, off_y))

        self._tk_img = ImageTk.PhotoImage(canvas_img)

        if self._canvas_image_id is None:
            self._canvas_image_id = self.video_canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        else:
            self.video_canvas.itemconfigure(self._canvas_image_id, image=self._tk_img)

        if self._canvas_text_id is not None:
            self.video_canvas.delete(self._canvas_text_id)
            self._canvas_text_id = None

        # ROI overlay only when user wants it
        if self._roi_mode == "SELECTING" and self._roi_start_canvas and self._roi_end_canvas:
            self._draw_roi_overlay()
        elif self._roi_mode == "DEFINED" and self.controller.roi_defined and self.controller.roi is not None:
            self._update_roi_overlay_from_controller()

    # -------------------------------------------------------------------------
    # Metrics refresh
    # -------------------------------------------------------------------------
    def _refresh_metrics(self) -> None:
        m = self.controller.metrics
        self.lbl_time.configure(text=f"Time: {m.elapsed_time_s:.2f} s")
        self.lbl_dist.configure(text=f"Distance: {m.total_distance_px:.2f} px")
        self.lbl_speed.configure(text=f"Speed: {m.instantaneous_speed_px_per_s:.2f} px/s")
        self.lbl_speed_avg.configure(text=f"Avg Speed: {m.average_speed_px_per_s:.2f} px/s")
        self.lbl_zone.configure(text=f"Zone: {m.current_zone}")

        self.lbl_freezing.configure(text=f"Freezing: {'ON' if m.freezing_active else 'off'} ({m.freezing_time_s:.1f} s)")
        self.lbl_grooming.configure(text=f"Grooming: {'ON' if m.grooming_active else 'off'} ({m.grooming_time_s:.1f} s)")

        self.after(200, self._refresh_metrics)
