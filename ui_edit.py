# IGNORE THIS MESSAGE ONLY FOR BRAINSTORMING PURPOSES
# Things to add - inverted images


"""
Sequence Editor UI — browse, reorder, delete/restore, and adjust tile positions
for the image sequence produced by sequence_v5.py.

Features:
  • Flip through images in sequence order with ←/→ keys or buttons.
  • Previous/next image transitions shown as semi-transparent overlays.
  • Delete images from the sequence into a queue bin.
  • Re-insert queued images at any position in the sequence.
  • Drag to move the tile crop rectangle on any image.
  • Scroll wheel to resize the tile.
  • Queue thumbnails with large preview on selection.
  • All changes persist back to sequence_order_v3.txt on save.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageEnhance
import copy


# ── Data Model ──────────────────────────────────────────────────────────────


class SequenceEntry:
    """One entry in the sequence."""

    __slots__ = ("filename", "tile_y", "tile_x", "tile_size", "ld_h", "ld_w")

    def __init__(self, filename, tile_y, tile_x, tile_size, ld_h, ld_w):
        self.filename = filename
        self.tile_y = tile_y
        self.tile_x = tile_x
        self.tile_size = tile_size
        self.ld_h = ld_h
        self.ld_w = ld_w

    def to_line(self):
        return f"{self.filename},{self.tile_y},{self.tile_x},{self.tile_size},{self.ld_h},{self.ld_w}"

    @staticmethod
    def from_line(line):
        parts = line.strip().split(",")
        if len(parts) < 6:
            return None
        return SequenceEntry(
            parts[0],
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
            int(parts[5]),
        )

    def clone(self):
        return SequenceEntry(
            self.filename,
            self.tile_y,
            self.tile_x,
            self.tile_size,
            self.ld_h,
            self.ld_w,
        )


def load_sequence(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = SequenceEntry.from_line(line)
            if e is not None:
                entries.append(e)
    return entries


def save_sequence(entries, path):
    with open(path, "w") as f:
        for e in entries:
            f.write(e.to_line() + "\n")


# ── Helpers ─────────────────────────────────────────────────────────────────


def find_photo(filename, foto_folder):
    """Resolve the actual photo path (try common extensions)."""
    base = os.path.splitext(filename)[0]
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".tif", ".bmp"):
        candidate = os.path.join(foto_folder, base + ext)
        if os.path.exists(candidate):
            return candidate
    candidate = os.path.join(foto_folder, filename)
    if os.path.exists(candidate):
        return candidate
    return None


def load_full_image(entry, foto_folder):
    """Load the full-resolution PIL image for an entry."""
    path = find_photo(entry.filename, foto_folder)
    if path is None:
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def crop_tile(pil_img, entry):
    """
    Crop the tile region from the full image, mapping low-res tile coords
    back to the actual image resolution.
    """
    actual_w, actual_h = pil_img.size
    scale = min(actual_h, actual_w) / min(entry.ld_h, entry.ld_w)

    ts = int(entry.tile_size * scale)
    ts = min(ts, actual_h, actual_w)
    sy = max(0, min(int(entry.tile_y * scale), actual_h - ts))
    sx = max(0, min(int(entry.tile_x * scale), actual_w - ts))

    return pil_img.crop((sx, sy, sx + ts, sy + ts))


def make_display_image(
    pil_img,
    entry,
    display_size,
    show_rect=True,
    rect_color=(0, 255, 0),
    rect_width=3,
):
    """
    Resize the full image to fit `display_size` and optionally draw
    the tile rectangle overlay.

    Returns (display_pil, ratio, offset_x, offset_y, ld_scale) so the
    caller can map mouse coords back to low-res image coords.
    """
    actual_w, actual_h = pil_img.size
    # Fit inside display_size keeping aspect ratio
    ds_w, ds_h = display_size
    ratio = min(ds_w / actual_w, ds_h / actual_h)
    new_w = int(actual_w * ratio)
    new_h = int(actual_h * ratio)
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

    # Center on a blank canvas
    canvas_img = Image.new("RGB", (ds_w, ds_h), (30, 30, 30))
    off_x = (ds_w - new_w) // 2
    off_y = (ds_h - new_h) // 2
    canvas_img.paste(resized, (off_x, off_y))

    # Map low-res tile coords → actual → display
    ld_scale = min(actual_h, actual_w) / min(entry.ld_h, entry.ld_w)

    if show_rect:
        draw = ImageDraw.Draw(canvas_img)
        ts_actual = int(entry.tile_size * ld_scale)
        ts_actual = min(ts_actual, actual_h, actual_w)
        sy = max(0, min(int(entry.tile_y * ld_scale), actual_h - ts_actual))
        sx = max(0, min(int(entry.tile_x * ld_scale), actual_w - ts_actual))

        # To display coords
        rx0 = off_x + int(sx * ratio)
        ry0 = off_y + int(sy * ratio)
        rx1 = off_x + int((sx + ts_actual) * ratio)
        ry1 = off_y + int((sy + ts_actual) * ratio)

        for w_off in range(rect_width):
            draw.rectangle(
                [rx0 - w_off, ry0 - w_off, rx1 + w_off, ry1 + w_off],
                outline=rect_color,
            )

    return canvas_img, ratio, off_x, off_y, ld_scale


def make_tile_overlay(pil_img, entry, overlay_size):
    """
    Create a cropped tile image resized to overlay_size.
    Returns PIL Image or None.
    """
    if pil_img is None:
        return None
    tile = crop_tile(pil_img, entry)
    if tile.size[0] == 0 or tile.size[1] == 0:
        return None
    return tile.resize((overlay_size, overlay_size), Image.LANCZOS)


def blend_overlay(base_canvas, overlay_pil, position, alpha=0.35):
    """
    Paste `overlay_pil` onto `base_canvas` at `position` with transparency.
    Modifies base_canvas in-place.
    """
    if overlay_pil is None:
        return
    overlay_rgba = overlay_pil.convert("RGBA")
    # Apply alpha
    r, g, b, a = overlay_rgba.split()
    a = a.point(lambda p: int(p * alpha))
    overlay_rgba = Image.merge("RGBA", (r, g, b, a))
    base_canvas_rgba = base_canvas.convert("RGBA")
    base_canvas_rgba.paste(overlay_rgba, position, overlay_rgba)
    # Copy back
    base_canvas.paste(base_canvas_rgba.convert("RGB"))


def make_transition_panel(
    cur_tile,
    neighbor_tile,
    panel_size,
    label_text,
    label_color,
    border_color,
    neighbor_alpha=0.45,
):
    """
    Build a transition preview panel: the current tile at full opacity with
    the neighbor tile overlaid semi-transparently on top.

    Returns a PIL RGB image of size (panel_size + border, panel_size + header).
    """
    border = 4
    header = 22
    total_w = panel_size + border
    total_h = panel_size + header

    panel = Image.new("RGB", (total_w, total_h), (30, 30, 30))

    # Start with current tile as base
    if cur_tile is not None:
        base = cur_tile.resize((panel_size, panel_size), Image.LANCZOS)
        panel.paste(base, (border // 2, header))
    else:
        # Grey placeholder
        draw_p = ImageDraw.Draw(panel)
        draw_p.rectangle(
            [border // 2, header, panel_size + border // 2, header + panel_size],
            fill=(50, 50, 50),
        )

    # Overlay the neighbor tile on top with transparency
    if neighbor_tile is not None:
        neighbor_resized = neighbor_tile.resize((panel_size, panel_size), Image.LANCZOS)
        neighbor_rgba = neighbor_resized.convert("RGBA")
        r, g, b, a = neighbor_rgba.split()
        a = a.point(lambda p: int(p * neighbor_alpha))
        neighbor_rgba = Image.merge("RGBA", (r, g, b, a))
        panel_rgba = panel.convert("RGBA")
        panel_rgba.paste(neighbor_rgba, (border // 2, header), neighbor_rgba)
        panel = panel_rgba.convert("RGB")

    # Draw label and border
    draw = ImageDraw.Draw(panel)
    draw.text((4, 2), label_text, fill=label_color)
    draw.rectangle(
        [0, header - 1, total_w - 1, total_h - 1],
        outline=border_color,
        width=2,
    )

    return panel


# ── Main UI ─────────────────────────────────────────────────────────────────


THUMB_SIZE = 48  # thumbnail pixel size in queue list


class SequenceEditorApp:
    DISPLAY_W = 900
    DISPLAY_H = 600
    OVERLAY_SIZE = 280  # larger transition preview panels
    OVERLAY_ALPHA = 0.7  # neighbor overlay opacity on top of current tile
    OVERLAY_PADDING = 12
    IMAGE_FRACTION = 0.55  # main image takes ~55% of canvas height
    TILE_RESIZE_STEP = 5  # low-res pixels per scroll tick
    TILE_MIN_SIZE = 10  # minimum tile size in low-res pixels
    QUEUE_PREVIEW_SIZE = 280  # large preview shown when queue item selected

    def __init__(self, root, sequence_path, foto_folder):
        self.root = root
        self.sequence_path = sequence_path
        self.foto_folder = foto_folder

        self.sequence = load_sequence(sequence_path)
        self.queue = []  # removed entries awaiting re-insertion
        self.current_idx = 0

        # Undo stack
        self.undo_stack = []

        # Drag state
        self._drag_start = None  # (mouse_x, mouse_y) at drag start
        self._drag_orig_tile = None  # (tile_y, tile_x) at drag start

        # Scroll-resize undo coalescing: track the tile_size before a scroll burst
        self._scroll_undo_pushed = False
        self._scroll_timer_id = None

        # Display transform cache (set by _render_current)
        self._display_ratio = 1.0
        self._display_off_x = 0
        self._display_off_y = 0
        self._display_ld_scale = 1.0
        self._image_area_y = 0  # top of the main image area on canvas
        self._image_area_h = 0  # height of the main image area

        # Image cache (LRU-ish)
        self._img_cache = {}
        self._cache_max = 20

        # Thumbnail cache for queue {filename: PhotoImage}
        self._thumb_cache = {}

        # Queue preview Tk image ref (prevent GC)
        self._queue_preview_tk = None

        self._dirty = False

        self._build_ui()
        self._render_current()

    # ── UI Construction ─────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("Sequence Editor")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(True, True)

        # ── Top toolbar ──
        toolbar = tk.Frame(self.root, bg="#2d2d2d", pady=4)
        toolbar.pack(fill=tk.X)

        btn_style = {
            "bg": "#3c3c3c",
            "fg": "white",
            "relief": "flat",
            "padx": 8,
            "pady": 3,
            "font": ("Segoe UI", 10),
        }

        self.btn_prev = tk.Button(
            toolbar, text="◀ Prev", command=self._go_prev, **btn_style
        )
        self.btn_prev.pack(side=tk.LEFT, padx=4)

        self.btn_next = tk.Button(
            toolbar, text="Next ▶", command=self._go_next, **btn_style
        )
        self.btn_next.pack(side=tk.LEFT, padx=4)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.btn_delete = tk.Button(
            toolbar,
            text="🗑 Remove from Sequence",
            command=self._delete_current,
            bg="#6b2020",
            fg="white",
            relief="flat",
            padx=8,
            pady=3,
            font=("Segoe UI", 10),
        )
        self.btn_delete.pack(side=tk.LEFT, padx=4)

        self.btn_insert = tk.Button(
            toolbar,
            text="📥 Insert from Queue Here",
            command=self._insert_from_queue,
            bg="#1b5e20",
            fg="white",
            relief="flat",
            padx=8,
            pady=3,
            font=("Segoe UI", 10),
        )
        self.btn_insert.pack(side=tk.LEFT, padx=4)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.btn_undo = tk.Button(
            toolbar, text="↩ Undo", command=self._undo, **btn_style
        )
        self.btn_undo.pack(side=tk.LEFT, padx=4)

        self.btn_save = tk.Button(
            toolbar,
            text="💾 Save",
            command=self._save,
            bg="#0d47a1",
            fg="white",
            relief="flat",
            padx=8,
            pady=3,
            font=("Segoe UI", 10, "bold"),
        )
        self.btn_save.pack(side=tk.RIGHT, padx=8)

        self.btn_reset_tile = tk.Button(
            toolbar, text="⊞ Center Tile", command=self._center_tile, **btn_style
        )
        self.btn_reset_tile.pack(side=tk.RIGHT, padx=4)

        # ── Tile size controls ──
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.RIGHT, fill=tk.Y, padx=8
        )

        self.btn_tile_bigger = tk.Button(
            toolbar, text="＋ Tile", command=self._tile_grow, **btn_style
        )
        self.btn_tile_bigger.pack(side=tk.RIGHT, padx=2)

        self.tile_size_var = tk.StringVar(value="size: —")
        self.tile_size_label = tk.Label(
            toolbar,
            textvariable=self.tile_size_var,
            bg="#2d2d2d",
            fg="#8f8",
            font=("Consolas", 10),
            padx=4,
        )
        self.tile_size_label.pack(side=tk.RIGHT, padx=2)

        self.btn_tile_smaller = tk.Button(
            toolbar, text="－ Tile", command=self._tile_shrink, **btn_style
        )
        self.btn_tile_smaller.pack(side=tk.RIGHT, padx=2)

        # ── Status bar ──
        self.status_var = tk.StringVar(value="")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#252525",
            fg="#aaa",
            anchor="w",
            font=("Consolas", 9),
            padx=6,
            pady=2,
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # ── Queue panel (right side drawer) ──
        queue_frame = tk.Frame(self.root, bg="#252525", width=260)
        queue_frame.pack(fill=tk.Y, side=tk.RIGHT)
        queue_frame.pack_propagate(False)

        tk.Label(
            queue_frame,
            text="Queue (removed)",
            bg="#252525",
            fg="#ccc",
            font=("Segoe UI", 11, "bold"),
            pady=6,
        ).pack(fill=tk.X)

        # ── Queue preview area (large image when selected) ──
        self.queue_preview_label = tk.Label(
            queue_frame,
            bg="#1a1a1a",
            relief="flat",
            text="Select an item\nto preview",
            fg="#555",
            font=("Segoe UI", 9),
            compound="top",
        )
        self.queue_preview_label.pack(fill=tk.X, padx=4, pady=(0, 4))

        self.queue_preview_name = tk.Label(
            queue_frame,
            bg="#252525",
            fg="#aaa",
            font=("Consolas", 8),
            text="",
            anchor="w",
        )
        self.queue_preview_name.pack(fill=tk.X, padx=6, pady=(0, 4))

        # ── Queue thumbnail list (Canvas-based for image support) ──
        list_frame = tk.Frame(queue_frame, bg="#1e1e1e")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        self.queue_canvas = tk.Canvas(
            list_frame,
            bg="#1e1e1e",
            highlightthickness=0,
            width=240,
        )
        self.queue_scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.queue_canvas.yview
        )
        self.queue_inner = tk.Frame(self.queue_canvas, bg="#1e1e1e")

        self.queue_inner.bind(
            "<Configure>",
            lambda e: self.queue_canvas.configure(
                scrollregion=self.queue_canvas.bbox("all")
            ),
        )
        self.queue_canvas.create_window((0, 0), window=self.queue_inner, anchor="nw")
        self.queue_canvas.configure(yscrollcommand=self.queue_scrollbar.set)

        self.queue_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.queue_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Enable mouse-wheel scrolling on the queue canvas
        self.queue_canvas.bind("<MouseWheel>", self._on_queue_scroll)
        self.queue_canvas.bind("<Button-4>", self._on_queue_scroll)
        self.queue_canvas.bind("<Button-5>", self._on_queue_scroll)

        # Track which queue row is selected
        self._queue_selected_idx = None
        self._queue_row_widgets = []  # list of (frame, label, thumb_label) per row

        queue_btn_frame = tk.Frame(queue_frame, bg="#252525")
        queue_btn_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        self.btn_queue_insert_before = tk.Button(
            queue_btn_frame,
            text="Insert Before",
            command=lambda: self._insert_from_queue(before=True),
            bg="#1b5e20",
            fg="white",
            relief="flat",
            font=("Segoe UI", 9),
        )
        self.btn_queue_insert_before.pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )

        self.btn_queue_insert_after = tk.Button(
            queue_btn_frame,
            text="Insert After",
            command=lambda: self._insert_from_queue(before=False),
            bg="#1b5e20",
            fg="white",
            relief="flat",
            font=("Segoe UI", 9),
        )
        self.btn_queue_insert_after.pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
        )

        # ── Main canvas ──
        canvas_frame = tk.Frame(self.root, bg="#1e1e1e")
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.DISPLAY_W,
            height=self.DISPLAY_H - 100,
            bg="#1a1a1a",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Mouse / keyboard bindings
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows scroll
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)  # Linux scroll down
        self.root.bind("<Left>", lambda e: self.move_tile_left())
        self.root.bind("<Right>", lambda e: self.move_tile_right())
        self.root.bind("<Up>", lambda e: self.move_tile_up())
        self.root.bind("<Down>", lambda e: self.move_tile_down())
        self.root.bind("<a>", lambda e: self._go_prev())
        self.root.bind("<d>", lambda e: self._go_next())
        self.root.bind("<Delete>", lambda e: self._delete_current())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Control-s>", lambda e: self._save())
        self.root.bind("<plus>", lambda e: self._tile_grow())
        self.root.bind("<equal>", lambda e: self._tile_grow())
        self.root.bind("<minus>", lambda e: self._tile_shrink())
        self.root.bind("<KP_Add>", lambda e: self._tile_grow())
        self.root.bind("<KP_Subtract>", lambda e: self._tile_shrink())

    # ── Image Cache ─────────────────────────────────────────────────────

    def _get_image(self, entry):
        """Load a PIL image for `entry`, with simple caching."""
        key = entry.filename
        if key in self._img_cache:
            return self._img_cache[key]
        img = load_full_image(entry, self.foto_folder)
        if img is not None:
            if len(self._img_cache) >= self._cache_max:
                oldest = next(iter(self._img_cache))
                del self._img_cache[oldest]
            self._img_cache[key] = img
        return img

    def _get_thumbnail(self, entry):
        """Return a Tk PhotoImage thumbnail for the queue list."""
        key = entry.filename
        if key in self._thumb_cache:
            return self._thumb_cache[key]

        pil_img = self._get_image(entry)
        if pil_img is None:
            return None

        # Crop tile then resize to thumbnail
        tile = crop_tile(pil_img, entry)
        if tile.size[0] == 0 or tile.size[1] == 0:
            return None
        tile_thumb = tile.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        tk_thumb = ImageTk.PhotoImage(tile_thumb)
        self._thumb_cache[key] = tk_thumb
        return tk_thumb

    # ── Queue Thumbnail List ────────────────────────────────────────────

    def _refresh_queue_list(self):
        """Rebuild the thumbnail queue list from self.queue."""
        # Destroy old rows
        for frame, _, _ in self._queue_row_widgets:
            frame.destroy()
        self._queue_row_widgets.clear()
        self._queue_selected_idx = None

        for i, entry in enumerate(self.queue):
            row = tk.Frame(self.queue_inner, bg="#1e1e1e", cursor="hand2")
            row.pack(fill=tk.X, padx=2, pady=1)

            thumb_tk = self._get_thumbnail(entry)

            thumb_label = tk.Label(
                row,
                bg="#1e1e1e",
                image=thumb_tk if thumb_tk else None,
                width=THUMB_SIZE,
                height=THUMB_SIZE,
            )
            if thumb_tk is None:
                thumb_label.configure(text="?", fg="#666", font=("Segoe UI", 9))
            thumb_label.pack(side=tk.LEFT, padx=(2, 6), pady=2)

            name_label = tk.Label(
                row,
                text=entry.filename,
                bg="#1e1e1e",
                fg="#ccc",
                font=("Consolas", 9),
                anchor="w",
            )
            name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Click handler — capture index
            idx = i
            for widget in (row, thumb_label, name_label):
                widget.bind(
                    "<Button-1>", lambda e, ii=idx: self._on_queue_row_click(ii)
                )

            self._queue_row_widgets.append((row, name_label, thumb_label))

        # Reset preview
        self._update_queue_preview()

    def _on_queue_row_click(self, idx):
        """Handle click on a queue row."""
        # Deselect old
        if self._queue_selected_idx is not None and self._queue_selected_idx < len(
            self._queue_row_widgets
        ):
            old_row, old_name, old_thumb = self._queue_row_widgets[
                self._queue_selected_idx
            ]
            old_row.configure(bg="#1e1e1e")
            old_name.configure(bg="#1e1e1e")
            old_thumb.configure(bg="#1e1e1e")

        self._queue_selected_idx = idx

        # Highlight new
        if idx < len(self._queue_row_widgets):
            row, name_l, thumb_l = self._queue_row_widgets[idx]
            row.configure(bg="#3c3c3c")
            name_l.configure(bg="#3c3c3c")
            thumb_l.configure(bg="#3c3c3c")

        self._update_queue_preview()

    def _update_queue_preview(self):
        """Show a large preview of the selected queue item."""
        if self._queue_selected_idx is None or self._queue_selected_idx >= len(
            self.queue
        ):
            self.queue_preview_label.configure(
                image="", text="Select an item\nto preview"
            )
            self.queue_preview_name.configure(text="")
            self._queue_preview_tk = None
            return

        entry = self.queue[self._queue_selected_idx]
        pil_img = self._get_image(entry)
        if pil_img is None:
            self.queue_preview_label.configure(
                image="", text=f"Cannot load\n{entry.filename}"
            )
            self.queue_preview_name.configure(text=entry.filename)
            self._queue_preview_tk = None
            return

        # Crop tile and resize to preview size
        tile = crop_tile(pil_img, entry)
        if tile.size[0] == 0 or tile.size[1] == 0:
            self.queue_preview_label.configure(image="", text="Bad tile")
            self._queue_preview_tk = None
            return

        preview_size = self.QUEUE_PREVIEW_SIZE
        # Fit inside preview_size keeping aspect
        tw, th = tile.size
        ratio = min(preview_size / tw, preview_size / th)
        pw, ph = int(tw * ratio), int(th * ratio)
        preview_pil = tile.resize((pw, ph), Image.LANCZOS)

        # Add a coloured border
        bordered = Image.new("RGB", (pw + 4, ph + 4), (200, 100, 50))
        bordered.paste(preview_pil, (2, 2))

        self._queue_preview_tk = ImageTk.PhotoImage(bordered)
        self.queue_preview_label.configure(image=self._queue_preview_tk, text="")
        self.queue_preview_name.configure(
            text=f"{entry.filename}  tile={entry.tile_size}"
        )

    def _on_queue_scroll(self, event):
        """Scroll the queue canvas."""
        if hasattr(event, "delta") and event.delta != 0:
            self.queue_canvas.yview_scroll(-event.delta // 120, "units")
        elif event.num == 4:
            self.queue_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.queue_canvas.yview_scroll(1, "units")

    def _get_queue_selection_index(self):
        """Return the currently selected queue index, or None."""
        return self._queue_selected_idx

    # ── Rendering ───────────────────────────────────────────────────────

    def _render_current(self):
        if not self.sequence:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.DISPLAY_W // 2,
                self.DISPLAY_H - 100 // 2,
                text="Sequence is empty.\nInsert images from the queue.",
                fill="#888",
                font=("Segoe UI", 16),
            )
            self._update_status()
            return

        idx = self.current_idx
        entry = self.sequence[idx]

        # Update canvas size from widget
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), self.DISPLAY_W)
        ch = max(self.canvas.winfo_height() - 100, self.DISPLAY_H - 100)

        pil_img = self._get_image(entry)
        if pil_img is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                cw // 2,
                ch // 2,
                text=f"Cannot load:\n{entry.filename}",
                fill="#f44",
                font=("Segoe UI", 14),
            )
            self._update_status()
            return

        # ── Layout: top region = main image (smaller), bottom = transition panels ──
        pad = self.OVERLAY_PADDING
        panel_header = 22
        # Compute panel size: fit two panels + current tile side by side
        # with padding, capped by available width and bottom region height
        bottom_h = int(ch * (1.0 - self.IMAGE_FRACTION))
        panel_size = min(
            self.OVERLAY_SIZE,
            bottom_h - panel_header - pad * 2,
            (cw - pad * 4) // 3 - 4,  # 3 panels across
        )
        panel_size = max(100, panel_size)

        image_area_h = ch - (panel_size + panel_header + pad * 3)
        image_area_h = max(200, image_area_h)
        self._image_area_y = 0
        self._image_area_h = image_area_h

        display_size = (cw, image_area_h)

        # Build the full canvas
        full_canvas = Image.new("RGB", (cw, ch), (26, 26, 26))

        # ── Main image with tile rect (top portion) ──
        img_canvas, ratio, off_x, off_y, ld_scale = make_display_image(
            pil_img,
            entry,
            display_size,
            show_rect=True,
            rect_color=(0, 255, 0),
            rect_width=2,
        )
        full_canvas.paste(img_canvas, (0, 0))

        self._display_ratio = ratio
        self._display_off_x = off_x
        self._display_off_y = off_y
        self._display_ld_scale = ld_scale

        # ── Bottom: transition panels ──
        panels_y = image_area_h + pad

        # Draw a subtle separator line
        draw_sep = ImageDraw.Draw(full_canvas)
        draw_sep.line(
            [(pad, image_area_h + 2), (cw - pad, image_area_h + 2)],
            fill=(60, 60, 60),
            width=1,
        )

        # Get current tile
        cur_tile = make_tile_overlay(pil_img, entry, panel_size)

        # ── Previous transition panel (left) ──
        prev_tile = None
        if idx > 0:
            prev_entry = self.sequence[idx - 1]
            prev_img = self._get_image(prev_entry)
            prev_tile = make_tile_overlay(prev_img, prev_entry, panel_size)

        prev_panel = make_transition_panel(
            cur_tile,
            prev_tile,
            panel_size,
            label_text="◀ PREV over CURRENT",
            label_color=(150, 150, 255),
            border_color=(100, 100, 255),
            neighbor_alpha=self.OVERLAY_ALPHA,
        )

        # ── Next transition panel (right) ──
        next_tile = None
        if idx < len(self.sequence) - 1:
            next_entry = self.sequence[idx + 1]
            next_img = self._get_image(next_entry)
            next_tile = make_tile_overlay(next_img, next_entry, panel_size)

        next_panel = make_transition_panel(
            cur_tile,
            next_tile,
            panel_size,
            label_text="NEXT ▶ over CURRENT",
            label_color=(150, 255, 150),
            border_color=(100, 255, 100),
            neighbor_alpha=self.OVERLAY_ALPHA,
        )

        # ── Current tile panel (center) ──
        cur_panel_w = panel_size + 4
        cur_panel_h = panel_size + panel_header
        cur_panel = Image.new("RGB", (cur_panel_w, cur_panel_h), (30, 30, 30))
        if cur_tile is not None:
            cur_resized = cur_tile.resize((panel_size, panel_size), Image.LANCZOS)
            cur_panel.paste(cur_resized, (2, panel_header))
        draw_cur = ImageDraw.Draw(cur_panel)
        draw_cur.text((4, 2), "CURRENT TILE", fill=(0, 255, 0))
        draw_cur.rectangle(
            [0, panel_header - 1, cur_panel_w - 1, cur_panel_h - 1],
            outline=(0, 200, 0),
            width=2,
        )

        # Position the three panels: evenly across the width
        total_panels_w = prev_panel.width + cur_panel_w + next_panel.width + pad * 4
        start_x = max(pad, (cw - total_panels_w) // 2 + pad)

        full_canvas.paste(prev_panel, (start_x, panels_y))
        center_x = start_x + prev_panel.width + pad
        full_canvas.paste(cur_panel, (center_x, panels_y))
        right_x = center_x + cur_panel_w + pad
        full_canvas.paste(next_panel, (right_x, panels_y))

        # Convert to Tk and display
        self._tk_image = ImageTk.PhotoImage(full_canvas)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_image)

        self._update_status()

    def _update_status(self):
        n = len(self.sequence)
        q = len(self.queue)
        if n == 0:
            self.status_var.set(f"Sequence: empty | Queue: {q} images")
            self.tile_size_var.set("size: —")
            return
        idx = self.current_idx
        entry = self.sequence[idx]
        dirty_marker = " *" if self._dirty else ""
        self.status_var.set(
            f"[{idx + 1}/{n}]  {entry.filename}  |  "
            f"tile=({entry.tile_y}, {entry.tile_x})  size={entry.tile_size}  "
            f"img=({entry.ld_h}×{entry.ld_w})  |  "
            f"Queue: {q}{dirty_marker}"
        )
        self.tile_size_var.set(f"size: {entry.tile_size}")

    def _refresh_queue_listbox(self):
        """Rebuild the queue panel (thumbnails + preview)."""
        self._refresh_queue_list()

    # ── Navigation ──────────────────────────────────────────────────────

    def _go_prev(self):
        if self.sequence and self.current_idx > 0:
            self.current_idx -= 1
            self._render_current()

    def _go_next(self):
        if self.sequence and self.current_idx < len(self.sequence) - 1:
            self.current_idx += 1
            self._render_current()

    # ── Delete / Insert ─────────────────────────────────────────────────

    def _push_undo(self):
        self.undo_stack.append(
            {
                "sequence": [e.clone() for e in self.sequence],
                "queue": [e.clone() for e in self.queue],
                "current_idx": self.current_idx,
            }
        )
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _delete_current(self):
        if not self.sequence:
            return
        self._push_undo()
        entry = self.sequence.pop(self.current_idx)
        self.queue.append(entry)
        self._dirty = True

        if self.current_idx >= len(self.sequence) and self.sequence:
            self.current_idx = len(self.sequence) - 1
        elif not self.sequence:
            self.current_idx = 0

        self._refresh_queue_listbox()
        self._render_current()

    def _insert_from_queue(self, before=True):
        """Insert the selected queue item before or after the current position."""
        idx = self._get_queue_selection_index()
        if idx is None or idx >= len(self.queue):
            messagebox.showinfo("Insert", "Select an image from the queue first.")
            return

        queue_idx = idx
        self._push_undo()
        entry = self.queue.pop(queue_idx)
        self._dirty = True

        # Invalidate thumbnail cache for this entry (tile may differ on re-insert)
        self._thumb_cache.pop(entry.filename, None)

        if not self.sequence:
            self.sequence.append(entry)
            self.current_idx = 0
        else:
            insert_pos = self.current_idx if before else self.current_idx + 1
            self.sequence.insert(insert_pos, entry)
            self.current_idx = insert_pos

        self._refresh_queue_listbox()
        self._render_current()

    def _undo(self):
        if not self.undo_stack:
            return
        state = self.undo_stack.pop()
        self.sequence = state["sequence"]
        self.queue = state["queue"]
        self.current_idx = state["current_idx"]
        self._dirty = True
        self._refresh_queue_listbox()
        self._render_current()

    # ── Tile Dragging ───────────────────────────────────────────────────

    def _display_to_ld(self, mx, my):
        """
        Convert a display (canvas) pixel coordinate to low-resolution
        edge-map coordinates.
        """
        if not self.sequence:
            return None, None
        entry = self.sequence[self.current_idx]
        pil_img = self._get_image(entry)
        if pil_img is None:
            return None, None

        ratio = self._display_ratio
        off_x = self._display_off_x
        off_y = self._display_off_y
        ld_scale = self._display_ld_scale

        # display → actual
        ax = (mx - off_x) / ratio
        ay = (my - off_y) / ratio

        # actual → low-res
        lx = ax / ld_scale
        ly = ay / ld_scale

        return lx, ly

    def _on_mouse_down(self, event):
        if not self.sequence:
            return
        # Only start drag if clicking in the main image area (top portion)
        if event.y > self._image_area_y + self._image_area_h:
            return
        entry = self.sequence[self.current_idx]
        self._drag_start = (event.x, event.y)
        self._drag_orig_tile = (entry.tile_y, entry.tile_x)

    def _on_mouse_drag(self, event):
        if self._drag_start is None or not self.sequence:
            return
        entry = self.sequence[self.current_idx]
        pil_img = self._get_image(entry)
        if pil_img is None:
            return

        ratio = self._display_ratio
        ld_scale = self._display_ld_scale

        # Mouse delta in display pixels → low-res pixels
        dx_display = event.x - self._drag_start[0]
        dy_display = event.y - self._drag_start[1]

        dx_ld = dx_display / (ratio * ld_scale)
        dy_ld = dy_display / (ratio * ld_scale)

        new_ty = int(self._drag_orig_tile[0] + dy_ld)
        new_tx = int(self._drag_orig_tile[1] + dx_ld)

        # Clamp to valid range
        new_ty = max(0, min(new_ty, entry.ld_h - entry.tile_size))
        new_tx = max(0, min(new_tx, entry.ld_w - entry.tile_size))

        entry.tile_y = new_ty
        entry.tile_x = new_tx
        self._dirty = True

        self._render_current()

    def move_tile(self, dx_ld, dy_ld):
        entry = self.sequence[self.current_idx]
        pil_img = self._get_image(entry)
        if pil_img is None:
            return

        new_ty = int(entry.tile_y + dy_ld)
        new_tx = int(entry.tile_x + dx_ld)

        # Clamp to valid range
        new_ty = max(0, min(new_ty, entry.ld_h - entry.tile_size))
        new_tx = max(0, min(new_tx, entry.ld_w - entry.tile_size))

        entry.tile_y = new_ty
        entry.tile_x = new_tx
        self._dirty = True

        self._render_current()

    def move_tile_right(self):
        self.move_tile(dx_ld=10, dy_ld=0)

    def move_tile_left(self):
        self.move_tile(dx_ld=-10, dy_ld=0)

    def move_tile_up(self):
        self.move_tile(dx_ld=0, dy_ld=-10)

    def move_tile_down(self):
        self.move_tile(dx_ld=0, dy_ld=10)

    def _on_mouse_up(self, event):
        if self._drag_start is not None and self.sequence:
            # Push undo only if position actually changed
            entry = self.sequence[self.current_idx]
            if self._drag_orig_tile and (
                entry.tile_y != self._drag_orig_tile[0]
                or entry.tile_x != self._drag_orig_tile[1]
            ):
                undo_entry = entry.clone()
                undo_entry.tile_y = self._drag_orig_tile[0]
                undo_entry.tile_x = self._drag_orig_tile[1]

                self.undo_stack.append(
                    {
                        "sequence": [
                            undo_entry.clone() if i == self.current_idx else e.clone()
                            for i, e in enumerate(self.sequence)
                        ],
                        "queue": [e.clone() for e in self.queue],
                        "current_idx": self.current_idx,
                    }
                )
                if len(self.undo_stack) > 50:
                    self.undo_stack.pop(0)

        self._drag_start = None
        self._drag_orig_tile = None

    # ── Tile Resize (scroll wheel + buttons) ────────────────────────────

    def _on_mouse_wheel(self, event):
        """Resize the tile with the scroll wheel (only over the main image area)."""
        if not self.sequence:
            return
        # Only resize if pointer is over the main image area
        if event.y > self._image_area_y + self._image_area_h:
            return

        # Determine scroll direction
        # Windows: event.delta (+120 up, -120 down)
        # Linux: event.num (4=up, 5=down)
        if hasattr(event, "delta") and event.delta != 0:
            ticks = event.delta // 120
        elif event.num == 4:
            ticks = 1
        elif event.num == 5:
            ticks = -1
        else:
            return

        # Push a single undo for the whole scroll burst
        if not self._scroll_undo_pushed:
            self._push_undo()
            self._scroll_undo_pushed = True

        # Reset the coalesce timer — after 500ms of no scrolling, allow a new undo push
        if self._scroll_timer_id is not None:
            self.root.after_cancel(self._scroll_timer_id)
        self._scroll_timer_id = self.root.after(500, self._scroll_undo_reset)

        self._resize_tile(ticks * self.TILE_RESIZE_STEP)

    def _scroll_undo_reset(self):
        """Called after scroll activity stops; allows a fresh undo push next time."""
        self._scroll_undo_pushed = False
        self._scroll_timer_id = None

    def _tile_grow(self):
        """Increase tile size by one step (button / key)."""
        if not self.sequence:
            return
        self._push_undo()
        self._resize_tile(self.TILE_RESIZE_STEP)

    def _tile_shrink(self):
        """Decrease tile size by one step (button / key)."""
        if not self.sequence:
            return
        self._push_undo()
        self._resize_tile(-self.TILE_RESIZE_STEP)

    def _resize_tile(self, delta):
        """
        Change the current tile_size by `delta` low-res pixels.
        Keeps the tile centred on its current centre, clamping to image bounds.
        """
        if not self.sequence:
            return
        entry = self.sequence[self.current_idx]

        old_size = entry.tile_size
        max_size = min(entry.ld_h, entry.ld_w)
        new_size = max(self.TILE_MIN_SIZE, min(old_size + delta, max_size))

        if new_size == old_size:
            return

        # Keep the tile centred on the same point
        old_center_y = entry.tile_y + old_size / 2.0
        old_center_x = entry.tile_x + old_size / 2.0

        new_ty = int(old_center_y - new_size / 2.0)
        new_tx = int(old_center_x - new_size / 2.0)

        # Clamp position so the tile stays inside the image
        new_ty = max(0, min(new_ty, entry.ld_h - new_size))
        new_tx = max(0, min(new_tx, entry.ld_w - new_size))

        entry.tile_size = new_size
        entry.tile_y = new_ty
        entry.tile_x = new_tx
        self._dirty = True

        self._render_current()

    # ── Tile Centering ──────────────────────────────────────────────────

    def _center_tile(self):
        if not self.sequence:
            return
        self._push_undo()
        entry = self.sequence[self.current_idx]
        entry.tile_y = max(0, (entry.ld_h - entry.tile_size) // 2)
        entry.tile_x = max(0, (entry.ld_w - entry.tile_size) // 2)
        self._dirty = True
        self._render_current()

    # ── Save ────────────────────────────────────────────────────────────

    def _save(self):
        save_sequence(self.sequence, self.sequence_path)
        self._dirty = False
        self._update_status()
        messagebox.showinfo("Saved", f"Sequence saved to:\n{self.sequence_path}")

    # ── Cleanup ─────────────────────────────────────────────────────────

    def on_closing(self):
        if self._dirty:
            ans = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes.\n\nSave before closing?",
            )
            if ans is None:
                return
            if ans:
                self._save()
        self.root.destroy()


# ── Entry Point ─────────────────────────────────────────────────────────────


def main():
    sequence_path = "sequence_order_v3.txt"
    foto_folder = "."

    if len(sys.argv) > 1:
        sequence_path = sys.argv[1]
    if len(sys.argv) > 2:
        foto_folder = sys.argv[2]

    if not os.path.exists(sequence_path):
        print(f"Sequence file not found: {sequence_path}")
        print("Usage: python ui_edit.py [sequence_file] [foto_folder]")
        sys.exit(1)

    root = tk.Tk()
    root.geometry("1200x1000")
    app = SequenceEditorApp(root, sequence_path, foto_folder)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
