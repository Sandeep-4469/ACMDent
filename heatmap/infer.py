import os
import json
import cv2
import numpy as np

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = "/data1/sandeep_projects/dental_analysis/augmented_dataset/images"
PRED_JSON = os.path.join(BASE_DIR, "heatmap_results_arc", "test_predictions.json")
OUT_DIR = os.path.join(BASE_DIR, "hrnet_results_arc", "visualizations")
CORRECTED_PRED_JSON = os.path.join(BASE_DIR, "hrnet_results_arc", "test_predictions_corrected.json")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CORRECTED_PRED_JSON), exist_ok=True)

# =========================
# LOAD JSON
# =========================
with open(PRED_JSON, "r") as f:
    predictions = json.load(f)

# =========================
# COLOR MAP (BGR)
# =========================
COLOR_SCALE = (0, 255, 255)      # Yellow
COLOR_INcisor = (0, 0, 255)      # Red
COLOR_ARC = (0, 255, 0)          # Green


def _line_center_x(pts, line_idx):
    p1 = pts[2 * line_idx]
    p2 = pts[2 * line_idx + 1]
    return 0.5 * (float(p1[0]) + float(p2[0]))


def _is_valid_line(pts, line_idx):
    p1 = pts[2 * line_idx]
    p2 = pts[2 * line_idx + 1]
    return min(p1[0], p1[1], p2[0], p2[1]) >= 0


def _swap_lines(pts, a, b):
    ia, ib = 2 * a, 2 * b
    pts[ia], pts[ib] = pts[ib], pts[ia]
    pts[ia + 1], pts[ib + 1] = pts[ib + 1], pts[ia + 1]


def _shift_line_x(pts, line_idx, delta):
    i = 2 * line_idx
    pts[i][0] += delta
    pts[i + 1][0] += delta


def _set_line_center_x(pts, line_idx, target_x):
    cur_x = _line_center_x(pts, line_idx)
    _shift_line_x(pts, line_idx, target_x - cur_x)


def _line_center_y(pts, line_idx):
    p1 = pts[2 * line_idx]
    p2 = pts[2 * line_idx + 1]
    return 0.5 * (float(p1[1]) + float(p2[1]))


def _get_line_points(pts, line_idx):
    i = 2 * line_idx
    return [pts[i][:], pts[i + 1][:]]


def _set_line_points(pts, line_idx, p1, p2):
    i = 2 * line_idx
    pts[i] = [float(p1[0]), float(p1[1])]
    pts[i + 1] = [float(p2[0]), float(p2[1])]


def _enforce_scale_rule(pts, max_dx=15.0, min_dy=30.0):
    scale_idx = 0
    if not _is_valid_line(pts, scale_idx):
        return

    p1, p2 = _get_line_points(pts, scale_idx)
    upper, lower = (p1, p2) if p1[1] <= p2[1] else (p2, p1)

    dx = float(lower[0] - upper[0])
    dy = float(lower[1] - upper[1])

    # Rule: horizontal crossing/spread should not exceed 15 px.
    if abs(dx) > max_dx:
        lower[0] = upper[0] + (max_dx if dx >= 0 else -max_dx)

    # Keep scale roughly vertical by ensuring a minimum vertical separation.
    if dy < min_dy:
        lower[1] = upper[1] + min_dy

    _set_line_points(pts, scale_idx, upper, lower)


def _enforce_arc_verticality(pts, line_idx):
    p1, p2 = _get_line_points(pts, line_idx)
    upper, lower = (p1, p2) if p1[1] <= p2[1] else (p2, p1)
    dy = max(1.0, float(lower[1] - upper[1]))
    dx = float(lower[0] - upper[0])
    max_abs_dx = max(8.0, 0.8 * dy)  # enforce |dx| < |dy| approximately
    if abs(dx) > max_abs_dx:
        lower[0] = upper[0] + (max_abs_dx if dx >= 0 else -max_abs_dx)
    _set_line_points(pts, line_idx, upper, lower)


def _arc_line_indices(is_maxilla):
    return (1, 2) if is_maxilla else (5, 6)


def _reassign_arc_endpoints(pts, left_arc_idx, right_arc_idx):
    points = []
    for line_idx in (left_arc_idx, right_arc_idx):
        p1, p2 = _get_line_points(pts, line_idx)
        points.append(p1)
        points.append(p2)

    # Split all 4 arc endpoints into top pair and bottom pair.
    by_y = sorted(points, key=lambda p: p[1])
    top = by_y[:2]
    bottom = by_y[2:]
    top_left, top_right = sorted(top, key=lambda p: p[0])
    bottom_left, bottom_right = sorted(bottom, key=lambda p: p[0])

    # Rebuild lines so each arc uses same-side top+bottom endpoint.
    _set_line_points(pts, left_arc_idx, top_left, bottom_left)
    _set_line_points(pts, right_arc_idx, top_right, bottom_right)


def _clamp_pts(pts, w, h):
    for p in pts:
        p[0] = int(max(0, min(w - 1, round(float(p[0])))))
        p[1] = int(max(0, min(h - 1, round(float(p[1])))))


def correct_arc_predictions(pts, img_w, img_h, is_maxilla):
    # Work on a mutable float copy.
    if len(pts) < 14:
        return pts
    fixed = [[float(x), float(y)] for x, y in pts]

    left_arc_idx, right_arc_idx = _arc_line_indices(is_maxilla)
    if not (_is_valid_line(fixed, left_arc_idx) and _is_valid_line(fixed, right_arc_idx)):
        _clamp_pts(fixed, img_w, img_h)
        return fixed

    # Do not use/modify scale; use image center as a stable, model-agnostic midline proxy.
    mid_x = 0.5 * float(img_w)

    # Apply scale rule first (do not alter other line semantics).
    _enforce_scale_rule(fixed, max_dx=15.0, min_dy=max(24.0, 0.06 * float(img_h)))

    # First fix crossed endpoint pairing between left/right arc lines.
    _reassign_arc_endpoints(fixed, left_arc_idx, right_arc_idx)

    # Enforce semantic ordering: left_arc must be left of right_arc.
    if _line_center_x(fixed, left_arc_idx) > _line_center_x(fixed, right_arc_idx):
        _swap_lines(fixed, left_arc_idx, right_arc_idx)

    left_x = _line_center_x(fixed, left_arc_idx)
    right_x = _line_center_x(fixed, right_arc_idx)

    # Enforce minimum arc separation to avoid collapse.
    min_sep = max(20.0, 0.14 * float(img_w))
    cur_sep = right_x - left_x
    if cur_sep < min_sep:
        delta = 0.5 * (min_sep - cur_sep)
        _shift_line_x(fixed, left_arc_idx, -delta)
        _shift_line_x(fixed, right_arc_idx, delta)

    left_x = _line_center_x(fixed, left_arc_idx)
    right_x = _line_center_x(fixed, right_arc_idx)

    # If both arcs fall on same side, keep likely-correct arc and move only the wrong one.
    if left_x >= mid_x and right_x >= mid_x:
        target_left_x = min(mid_x - 0.5 * min_sep, right_x - min_sep)
        _set_line_center_x(fixed, left_arc_idx, target_left_x)
    elif left_x <= mid_x and right_x <= mid_x:
        target_right_x = max(mid_x + 0.5 * min_sep, left_x + min_sep)
        _set_line_center_x(fixed, right_arc_idx, target_right_x)

    # Final semantic ordering.
    if _line_center_x(fixed, left_arc_idx) > _line_center_x(fixed, right_arc_idx):
        _swap_lines(fixed, left_arc_idx, right_arc_idx)

    # Strong side constraint: each arc center must stay on its half.
    # Use incisor side anchors for mandible when available.
    left_anchor = mid_x - 0.2 * float(img_w)
    right_anchor = mid_x + 0.2 * float(img_w)
    if is_maxilla:
        if _is_valid_line(fixed, left_arc_idx):
            left_anchor = min(left_anchor, _line_center_x(fixed, left_arc_idx))
        if _is_valid_line(fixed, right_arc_idx):
            right_anchor = max(right_anchor, _line_center_x(fixed, right_arc_idx))
    else:
        if _is_valid_line(fixed, 1):
            left_anchor = min(left_anchor, _line_center_x(fixed, 1))
        if _is_valid_line(fixed, 4):
            right_anchor = max(right_anchor, _line_center_x(fixed, 4))

    max_shift = 0.28 * float(img_w)
    lx = _line_center_x(fixed, left_arc_idx)
    rx = _line_center_x(fixed, right_arc_idx)
    half_margin = max(10.0, 0.05 * float(img_w))

    if lx > mid_x - half_margin:
        target = min(left_anchor, mid_x - half_margin)
        _set_line_center_x(fixed, left_arc_idx, lx + np.clip(target - lx, -max_shift, max_shift))
    if rx < mid_x + half_margin:
        target = max(right_anchor, mid_x + half_margin)
        _set_line_center_x(fixed, right_arc_idx, rx + np.clip(target - rx, -max_shift, max_shift))

    # Keep arc tops above their bottoms after shifting.
    for arc_idx in (left_arc_idx, right_arc_idx):
        p1, p2 = _get_line_points(fixed, arc_idx)
        if p1[1] > p2[1]:
            _set_line_points(fixed, arc_idx, p2, p1)

    # Extra guard: right arc should not be left of left arc.
    if _line_center_x(fixed, left_arc_idx) > _line_center_x(fixed, right_arc_idx):
        _swap_lines(fixed, left_arc_idx, right_arc_idx)

    # Encourage arc tops to be sufficiently separated horizontally.
    ltop, _ = _get_line_points(fixed, left_arc_idx)
    rtop, _ = _get_line_points(fixed, right_arc_idx)
    top_sep_min = max(16.0, 0.1 * float(img_w))
    top_sep = float(rtop[0] - ltop[0])
    if top_sep < top_sep_min:
        push = 0.5 * (top_sep_min - top_sep)
        _shift_line_x(fixed, left_arc_idx, -push)
        _shift_line_x(fixed, right_arc_idx, push)

    # Enforce same-arc geometry: endpoints should be mostly vertical.
    _enforce_arc_verticality(fixed, left_arc_idx)
    _enforce_arc_verticality(fixed, right_arc_idx)

    _clamp_pts(fixed, img_w, img_h)
    return fixed


# =========================
# PROCESS EACH IMAGE
# =========================
corrected_predictions = {}
for name, data in predictions.items():

    img_path = os.path.join(IMG_DIR, f"{name}.jpg")
    img = cv2.imread(img_path)

    if img is None:
        print(f"Image not found: {name}")
        continue

    pts = data["predicted_keypoints"]
    is_maxilla = "maxilla" in name.lower()
    pts = correct_arc_predictions(pts, img.shape[1], img.shape[0], is_maxilla=is_maxilla)
    corrected_predictions[name] = {
        **data,
        "predicted_keypoints": pts,
    }
    n = len(pts)

    MANDIBLE_LINE_NAMES = [
        "scale",
        "incisor1",
        "incisor2",
        "incisor3",
        "incisor4",
        "left_arc",
        "right_arc"
    ]

    MAXILLA_ALLOWED = {
        0: "scale",
        1: "left_arc",
        2: "right_arc"
    }

    for i in range(0, n, 2):

        line_idx = i // 2

        if is_maxilla:
            if line_idx not in MAXILLA_ALLOWED:
                continue
            label = MAXILLA_ALLOWED[line_idx]
        else:
            if line_idx >= len(MANDIBLE_LINE_NAMES):
                continue
            label = MANDIBLE_LINE_NAMES[line_idx]

        x1, y1 = map(int, pts[i])
        x2, y2 = map(int, pts[i+1])

        if label == "scale":
            color = COLOR_SCALE
        elif "incisor" in label:
            color = COLOR_INcisor
        else:
            color = COLOR_ARC

        cv2.line(img, (x1, y1), (x2, y2), color, 3)
        cv2.circle(img, (x1, y1), 6, (255, 0, 0), -1)
        cv2.circle(img, (x2, y2), 6, (255, 0, 0), -1)

        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    save_path = os.path.join(OUT_DIR, f"{name}.jpg")
    cv2.imwrite(save_path, img)

print("✅ All visualizations saved to:", OUT_DIR)
with open(CORRECTED_PRED_JSON, "w") as f:
    json.dump(corrected_predictions, f, indent=2)
print("✅ Corrected predictions saved to:", CORRECTED_PRED_JSON)
