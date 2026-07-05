#!/usr/bin/env python3
"""
Generate docs/hero.png - the README image.

Left panel is REAL output: YOLOv8-L run on CUDA over the bundled bus.jpg, drawn
with the model's own annotator. Right panel is a schematic of the FaceNet
identity stage (MTCNN + InceptionResnetV1, Euclidean distance vs. the 0.8
threshold) - rendered without using the real reference photo in known_faces/.

Run:  python make_hero.py     (needs ultralytics + matplotlib + a CUDA GPU)
"""
import os
from collections import Counter

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
from ultralytics import YOLO

BG, INK, DIM = "#0b1017", "#e6ebf2", "#8493a6"
OK, BAD, ACC = "#28c76f", "#ff4d4d", "#3fa7ff"

# ---------- real YOLOv8 detection ----------
model = YOLO("yolov8l.pt")
res = model.predict("bus.jpg", device="cuda", conf=0.5, verbose=False)[0]
counts = Counter(res.names[int(c)] for c in res.boxes.cls)
annotated = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
fig = plt.figure(figsize=(13, 6.8), facecolor=BG)
fig.text(0.04, 0.945, "CUDA RTSP TRACKER  ·  DUAL-MODEL VISION NODE",
         fontsize=15.5, fontweight="bold")
fig.text(0.04, 0.900, "mobile RTSP / IP stream  →  RTX 3080 (CUDA)  →  YOLOv8 object tracking  +  "
                      "FaceNet identity  →  live overlay  ·  zero-cloud",
         fontsize=9.3, color=DIM)

# ---------- left: real detection ----------
axi = fig.add_axes([0.04, 0.08, 0.34, 0.78])
axi.imshow(annotated)
axi.set_xticks([]); axi.set_yticks([])
for s in axi.spines.values():
    s.set_edgecolor("#243247"); s.set_linewidth(1.4)
det_str = ", ".join(f"{v} {k}" for k, v in counts.most_common())
axi.set_title(f"YOLOv8-L · CUDA · {len(res.boxes)} detections", fontsize=9.5,
              color=DIM, loc="left", pad=6)
axi.text(0.5, -0.045, det_str, transform=axi.transAxes, ha="center",
         fontsize=9, color=INK)


def draw_face(ax, x, y, w, h, color, name, dist, known):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                 facecolor="#0a0f18", edgecolor=color, lw=2, transform=ax.transAxes))
    # stylized head (no real photo)
    cx = x + w / 2
    ax.add_patch(Circle((cx, y + h * 0.62), h * 0.17, facecolor="#2a3446",
                 edgecolor=color, lw=1.3, transform=ax.transAxes))
    ax.add_patch(Polygon([[cx - w * 0.22, y + h * 0.30], [cx + w * 0.22, y + h * 0.30],
                          [cx + w * 0.30, y + h * 0.44], [cx - w * 0.30, y + h * 0.44]],
                 closed=True, facecolor="#2a3446", edgecolor=color, lw=1.3, transform=ax.transAxes))
    ax.text(cx, y + h * 0.17, name, ha="center", fontsize=10, fontweight="bold",
            color=color, transform=ax.transAxes)
    ax.text(cx, y + h * 0.05, f"d = {dist:.2f}  {'✓' if known else '✗'}", ha="center",
            fontsize=8.5, color=DIM, transform=ax.transAxes)
    tag = "KNOWN" if known else "UNKNOWN"
    ax.text(x + 0.02, y + h - 0.05, tag, fontsize=7.5, fontweight="bold", color=color,
            transform=ax.transAxes, va="top")


# ---------- right-top: FaceNet identity ----------
axf = fig.add_axes([0.42, 0.40, 0.54, 0.46]); axf.axis("off")
axf.set_xlim(0, 1); axf.set_ylim(0, 1)
axf.text(0.0, 1.0, "FaceNet identity  —  MTCNN detect → InceptionResnetV1 (VGGFace2) embed",
         fontsize=9.5, color=DIM, fontweight="bold", transform=axf.transAxes)
draw_face(axf, 0.03, 0.14, 0.40, 0.72, OK, "me", 0.42, True)
draw_face(axf, 0.50, 0.14, 0.40, 0.72, BAD, "Unknown", 1.31, False)
axf.text(0.5, 0.03, "match if Euclidean distance < 0.80 threshold",
         ha="center", fontsize=8.6, color=INK, style="italic", transform=axf.transAxes)

# ---------- right-bottom: spec strip ----------
axs = fig.add_axes([0.42, 0.08, 0.54, 0.24], facecolor="#0a0f18"); axs.axis("off")
axs.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03",
              transform=axs.transAxes, facecolor="#0a0f18", edgecolor="#243247", lw=1.4))
specs = [
    ("objects", "YOLOv8-L", ACC),
    ("faces", "MTCNN + ResNet", OK),
    ("accel", "RTX 3080 · CUDA 12.1", "#ffb020"),
    ("stream", "local RTSP / IP cam", INK),
    ("privacy", "100% on-device", OK),
    ("threshold", "d < 0.80", INK),
]
for i, (k, v, col) in enumerate(specs):
    cx = 0.09 + (i % 3) * 0.31
    cy = 0.68 - (i // 3) * 0.42
    axs.text(cx, cy, k.upper(), fontsize=7.5, color=DIM, transform=axs.transAxes, ha="left")
    axs.text(cx, cy - 0.16, v, fontsize=9.5, color=col, fontweight="bold",
             transform=axs.transAxes, ha="left")

fig.text(0.04, 0.025, "Left: real YOLOv8-L / CUDA output on bundled bus.jpg. Right: FaceNet identity "
                      "stage (schematic — no reference photo shown). Regenerate: python make_hero.py",
         fontsize=8, color=DIM)

os.makedirs("docs", exist_ok=True)
fig.savefig("docs/hero.png", dpi=140, facecolor=BG)
print(f"[+] wrote docs/hero.png  ({len(res.boxes)} detections: {dict(counts)})")
