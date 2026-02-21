import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv

IMG_SIZE = 512
HEATMAP_SIZE = 128
NUM_KPS = 14

BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20
GEOM_WARMUP_EPOCHS = 25
MAX_LAMBDA_GEOM = 0.0
SCALE_MAX_DX = 15.0
ARC_MIN_SEP = 40.0
ARC_MARGIN = 3.0
ARC_POINT_WEIGHT = 1.2
SCALE_POINT_WEIGHT = 1.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_DIR = "/data1/sandeep_projects/dental_analysis/augmented_dataset/images"
ANN_FILE = "/data1/sandeep_projects/dental_analysis/augmented_dataset/combined_annotations_augmented.json"

SAVE_DIR = "heatmap_results_arc"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_CSV = os.path.join(SAVE_DIR, "training_log.csv")

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "heatmap_best.pth")
PRED_JSON = os.path.join(SAVE_DIR, "test_predictions.json")
INIT_MODEL_PATH = os.path.join("heatmap_results_1", "heatmap_best.pth")

class HeatmapDentalDataset(Dataset):
    def __init__(self, keys, data_dict, img_dir):
        self.keys = keys
        self.data = data_dict
        self.img_dir = img_dir

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.keys)

    def _generate_heatmap(self, pts):
        heatmaps = np.zeros((NUM_KPS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        sigma = 3

        xx, yy = np.meshgrid(np.arange(HEATMAP_SIZE), np.arange(HEATMAP_SIZE))

        for i, (x, y) in enumerate(pts):

            if x < 0 or y < 0:
                continue

            x = int(x * HEATMAP_SIZE / IMG_SIZE)
            y = int(y * HEATMAP_SIZE / IMG_SIZE)

            heatmaps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))

        return heatmaps

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]

        img_path = os.path.join(self.img_dir, f"{key}.jpg")
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lines = item["lines"]  # ordered lines

        pts = []
        for line in lines:
            pts.append(line[0])
            pts.append(line[1])

        # Pad to 14 keypoints
        while len(pts) < NUM_KPS:
            pts.append([-1, -1])

        pts = pts[:NUM_KPS]

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        scale_x = IMG_SIZE / w
        scale_y = IMG_SIZE / h

        scaled_pts = []
        for x, y in pts:
            if x < 0:
                scaled_pts.append([-1, -1])
            else:
                scaled_pts.append([x * scale_x, y * scale_y])

        heatmaps = self._generate_heatmap(scaled_pts)
        img = self.normalize(img)

        return img, torch.tensor(heatmaps), torch.tensor(scaled_pts), key


class HeatmapNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.deconv = nn.Sequential(
        nn.ConvTranspose2d(2048, 256, 4, 2, 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.ConvTranspose2d(256, 256, 4, 2, 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),

        nn.ConvTranspose2d(256, 256, 4, 2, 1), 
        nn.BatchNorm2d(256),
        nn.ReLU(),
)


        self.final_layer = nn.Conv2d(256, NUM_KPS, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv(x)
        x = self.final_layer(x)
        return x


def decode_heatmap(hm):
    hm = hm.cpu().numpy()
    coords = []
    for i in range(NUM_KPS):
        y, x = np.unravel_index(np.argmax(hm[i]), hm[i].shape)
        coords.append([x, y])
    return coords


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
    if abs(dx) > max_dx:
        lower[0] = upper[0] + (max_dx if dx >= 0 else -max_dx)
    if dy < min_dy:
        lower[1] = upper[1] + min_dy
    _set_line_points(pts, scale_idx, upper, lower)


def _enforce_arc_verticality(pts, line_idx):
    p1, p2 = _get_line_points(pts, line_idx)
    upper, lower = (p1, p2) if p1[1] <= p2[1] else (p2, p1)
    dy = max(1.0, float(lower[1] - upper[1]))
    dx = float(lower[0] - upper[0])
    max_abs_dx = max(8.0, 0.8 * dy)
    if abs(dx) > max_abs_dx:
        lower[0] = upper[0] + (max_abs_dx if dx >= 0 else -max_abs_dx)
    _set_line_points(pts, line_idx, upper, lower)


def _reassign_arc_endpoints(pts, left_arc_idx, right_arc_idx):
    points = []
    for line_idx in (left_arc_idx, right_arc_idx):
        p1, p2 = _get_line_points(pts, line_idx)
        points.append(p1)
        points.append(p2)

    by_y = sorted(points, key=lambda p: p[1])
    top = by_y[:2]
    bottom = by_y[2:]
    top_left, top_right = sorted(top, key=lambda p: p[0])
    bottom_left, bottom_right = sorted(bottom, key=lambda p: p[0])
    _set_line_points(pts, left_arc_idx, top_left, bottom_left)
    _set_line_points(pts, right_arc_idx, top_right, bottom_right)


def _clamp_pts(pts, w, h):
    for p in pts:
        p[0] = int(max(0, min(w - 1, round(float(p[0])))))
        p[1] = int(max(0, min(h - 1, round(float(p[1])))))


def correct_predictions_for_export(pts, img_w=IMG_SIZE, img_h=IMG_SIZE):
    if len(pts) < NUM_KPS:
        return pts
    fixed = [[float(x), float(y)] for x, y in pts]
    valid_kps = sum(1 for p in fixed if p[0] >= 0 and p[1] >= 0)

    _enforce_scale_rule(fixed, max_dx=SCALE_MAX_DX, min_dy=max(24.0, 0.06 * float(img_h)))
    mid_x = 0.5 * float(img_w)
    if valid_kps == 6:
        left_arc_idx, right_arc_idx = 1, 2
    else:
        left_arc_idx, right_arc_idx = 5, 6

    if _is_valid_line(fixed, left_arc_idx) and _is_valid_line(fixed, right_arc_idx):
        _reassign_arc_endpoints(fixed, left_arc_idx, right_arc_idx)
        if _line_center_x(fixed, left_arc_idx) > _line_center_x(fixed, right_arc_idx):
            _swap_lines(fixed, left_arc_idx, right_arc_idx)

        min_sep = max(20.0, 0.14 * float(img_w))
        left_x = _line_center_x(fixed, left_arc_idx)
        right_x = _line_center_x(fixed, right_arc_idx)
        if (right_x - left_x) < min_sep:
            delta = 0.5 * (min_sep - (right_x - left_x))
            _shift_line_x(fixed, left_arc_idx, -delta)
            _shift_line_x(fixed, right_arc_idx, delta)

        left_anchor = mid_x - 0.2 * float(img_w)
        right_anchor = mid_x + 0.2 * float(img_w)
        if valid_kps != 6:
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

        ltop, _ = _get_line_points(fixed, left_arc_idx)
        rtop, _ = _get_line_points(fixed, right_arc_idx)
        top_sep_min = max(16.0, 0.1 * float(img_w))
        top_sep = float(rtop[0] - ltop[0])
        if top_sep < top_sep_min:
            push = 0.5 * (top_sep_min - top_sep)
            _shift_line_x(fixed, left_arc_idx, -push)
            _shift_line_x(fixed, right_arc_idx, push)

        _enforce_arc_verticality(fixed, left_arc_idx)
        _enforce_arc_verticality(fixed, right_arc_idx)
        if _line_center_x(fixed, left_arc_idx) > _line_center_x(fixed, right_arc_idx):
            _swap_lines(fixed, left_arc_idx, right_arc_idx)

    _clamp_pts(fixed, img_w, img_h)
    return fixed


def _channel_weights(device):
    w = torch.ones(NUM_KPS, device=device)
    w[0:2] = SCALE_POINT_WEIGHT
    w[2:6] = ARC_POINT_WEIGHT
    w[10:14] = ARC_POINT_WEIGHT
    return w.view(1, NUM_KPS, 1, 1)


def _soft_argmax_2d(logits):
    b, c, h, w = logits.shape
    flat = logits.view(b, c, -1)
    prob = torch.softmax(flat, dim=-1)

    ys = torch.arange(h, device=logits.device, dtype=logits.dtype).view(1, 1, h, 1).expand(b, c, h, w)
    xs = torch.arange(w, device=logits.device, dtype=logits.dtype).view(1, 1, 1, w).expand(b, c, h, w)
    ys = ys.reshape(b, c, -1)
    xs = xs.reshape(b, c, -1)

    exp_x = torch.sum(prob * xs, dim=-1)
    exp_y = torch.sum(prob * ys, dim=-1)
    return torch.stack([exp_x, exp_y], dim=-1)  # [B, K, 2]


def _geometry_loss(pred_logits, gt_pts):
    pred_xy_hm = _soft_argmax_2d(pred_logits)
    scale = pred_logits.new_tensor(float(IMG_SIZE) / float(HEATMAP_SIZE))
    pred_xy = pred_xy_hm * scale
    total = pred_logits.new_tensor(0.0)
    count = 0

    for b in range(pred_xy.shape[0]):
        gt = gt_pts[b]
        valid = gt[:, 0] >= 0
        valid_kps = int(valid.sum().item())
        if valid_kps < 6:
            continue

        # scale line constraint
        if bool(valid[0] and valid[1]):
            dx_scale = pred_xy[b, 1, 0] - pred_xy[b, 0, 0]
            total = total + torch.relu(torch.abs(dx_scale) - SCALE_MAX_DX)
            count += 1

        if valid_kps == 6:
            left_line, right_line = 1, 2
        else:
            left_line, right_line = 5, 6

        li0, li1 = 2 * left_line, 2 * left_line + 1
        ri0, ri1 = 2 * right_line, 2 * right_line + 1
        if not bool(valid[li0] and valid[li1] and valid[ri0] and valid[ri1]):
            continue

        lcx = 0.5 * (pred_xy[b, li0, 0] + pred_xy[b, li1, 0])
        rcx = 0.5 * (pred_xy[b, ri0, 0] + pred_xy[b, ri1, 0])
        ldx = pred_xy[b, li1, 0] - pred_xy[b, li0, 0]
        ldy = pred_xy[b, li1, 1] - pred_xy[b, li0, 1]
        rdx = pred_xy[b, ri1, 0] - pred_xy[b, ri0, 0]
        rdy = pred_xy[b, ri1, 1] - pred_xy[b, ri0, 1]
        arc_sep = rcx - lcx

        total = total + torch.relu(lcx + ARC_MARGIN - rcx)
        total = total + torch.relu(ARC_MIN_SEP - arc_sep)
        total = total + torch.relu(torch.abs(ldx) - torch.abs(ldy))
        total = total + torch.relu(torch.abs(rdx) - torch.abs(rdy))
        count += 4

    if count == 0:
        return pred_logits.new_tensor(0.0)
    return total / float(count)


def train():

    with open(ANN_FILE) as f:
        data_dict = json.load(f)

    keys = list(data_dict.keys())
    train_keys, test_keys = train_test_split(keys, test_size=0.2, random_state=42)

    train_dataset = HeatmapDentalDataset(train_keys, data_dict, IMG_DIR)
    test_dataset = HeatmapDentalDataset(test_keys, data_dict, IMG_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = HeatmapNet().to(DEVICE)
    if os.path.exists(INIT_MODEL_PATH):
        try:
            state = torch.load(INIT_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
            print(f"Loaded initialization weights from: {INIT_MODEL_PATH}")
        except Exception as e:
            print(f"Could not load init weights ({e}). Training from scratch.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_loss = float("inf")
    patience_counter = 0
    ch_w = _channel_weights(DEVICE)

    for epoch in range(EPOCHS):
        # Stage-1: pure heatmap learning. Stage-2: slowly add geometry regularization.
        if epoch < GEOM_WARMUP_EPOCHS:
            lambda_geom = 0.0
        else:
            ramp = min(1.0, (epoch - GEOM_WARMUP_EPOCHS + 1) / 20.0)
            lambda_geom = MAX_LAMBDA_GEOM * ramp

        model.train()
        train_loss = 0

        for imgs, heatmaps, gt_pts, _ in train_loader:
            imgs = imgs.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)
            gt_pts = gt_pts.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)

            # MASK INVALID HEATMAPS
            diff = (preds - heatmaps) ** 2   # [B, 14, 128, 128]

            # valid keypoints mask
            valid_mask = (heatmaps.sum(dim=(2,3)) > 0).float()   # [B,14]

            # expand to heatmap size
            valid_mask = valid_mask.unsqueeze(2).unsqueeze(3)    # [B,14,1,1]
            valid_mask = valid_mask.expand_as(diff)              # [B,14,128,128]

            # apply mask
            diff = diff * valid_mask * ch_w

            # avoid dividing by padded elements
            hm_loss = diff.sum() / ((valid_mask * ch_w).sum() + 1e-6)
            geom_loss = _geometry_loss(preds, gt_pts)
            loss = hm_loss + lambda_geom * geom_loss

            loss = loss.mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALIDATION
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, heatmaps, gt_pts, _ in test_loader:
                imgs = imgs.to(DEVICE)
                heatmaps = heatmaps.to(DEVICE)
                gt_pts = gt_pts.to(DEVICE)

                preds = model(imgs)

                valid_mask = (heatmaps.sum(dim=(2,3)) > 0).float().unsqueeze(-1).unsqueeze(-1)
                valid_mask = valid_mask.expand_as(preds)
                diff = ((preds - heatmaps) ** 2) * valid_mask * ch_w
                hm_loss = diff.sum() / ((valid_mask * ch_w).sum() + 1e-6)
                geom_loss = _geometry_loss(preds, gt_pts)
                loss = hm_loss + lambda_geom * geom_loss

                val_loss += loss.item()

        val_loss /= len(test_loader)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch} | lambda_geom {lambda_geom:.4f} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("✅ Saved Best Model")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("🛑 Early stopping")
            break

    print("Training Complete")
    evaluate(test_loader)



def evaluate(loader):

    model = HeatmapNet().to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    LINE_NAMES_MANDIBLE = [
        "scale","incisor1","incisor2","incisor3",
        "incisor4","left_arc","right_arc"
    ]

    LINE_NAMES_MAXILLA = [
        "scale","left_arc","right_arc"
    ]

    line_errors_dict = {}
    predictions = {}

    with torch.no_grad():
        for imgs, _, gt_pts, keys in tqdm(loader):

            imgs = imgs.to(DEVICE)
            preds = model(imgs)

            for i in range(len(imgs)):

                pred_coords = decode_heatmap(preds[i])

                pred_pts = []
                for x, y in pred_coords:
                    px = x * IMG_SIZE / HEATMAP_SIZE
                    py = y * IMG_SIZE / HEATMAP_SIZE
                    pred_pts.append([px, py])
                pred_pts = correct_predictions_for_export(pred_pts, img_w=IMG_SIZE, img_h=IMG_SIZE)

                gt = gt_pts[i].numpy()
                valid_kps = np.sum(gt[:,0] >= 0)

                for j in range(0, NUM_KPS, 2):

                    if gt[j][0] < 0:
                        continue

                    p1 = np.array(pred_pts[j])
                    p2 = np.array(pred_pts[j+1])
                    g1 = gt[j]
                    g2 = gt[j+1]

                    pred_len = np.linalg.norm(p1 - p2)
                    gt_len = np.linalg.norm(g1 - g2)

                    error = abs(pred_len - gt_len)
                    line_idx = j // 2

                    if valid_kps == 6:
                        name = LINE_NAMES_MAXILLA[line_idx]
                    else:
                        name = LINE_NAMES_MANDIBLE[line_idx]

                    if name not in line_errors_dict:
                        line_errors_dict[name] = []

                    line_errors_dict[name].append(error)

                predictions[keys[i]] = {
                    "predicted_keypoints": pred_pts
                }

    total = []

    for name, errors in line_errors_dict.items():
        mean_err = np.mean(errors)
        total.extend(errors)
        print(f"{name}: {mean_err:.3f}")

    print(f"Overall Mean Line Pixel Error: {np.mean(total):.3f}")

    with open(PRED_JSON, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"✅ Predictions saved to {PRED_JSON}")


if __name__ == "__main__":
    train()
