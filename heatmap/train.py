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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_DIR = "/data1/sandeep_projects/dental_analysis/augmented_dataset/images"
ANN_FILE = "/data1/sandeep_projects/dental_analysis/augmented_dataset/combined_annotations_augmented.json"

SAVE_DIR = "heatmap_results_1"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_CSV = os.path.join(SAVE_DIR, "training_log.csv")

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "heatmap_best.pth")
PRED_JSON = os.path.join(SAVE_DIR, "test_predictions.json")

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


# =========================================================
# MODEL
# =========================================================
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0

        for imgs, heatmaps, _, _ in train_loader:
            imgs = imgs.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)

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
            diff = diff * valid_mask

            # avoid dividing by padded elements
            loss = diff.sum() / (valid_mask.sum() + 1e-6)

            loss = loss.mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALIDATION
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, heatmaps, _, _ in test_loader:
                imgs = imgs.to(DEVICE)
                heatmaps = heatmaps.to(DEVICE)

                preds = model(imgs)

                valid_mask = (heatmaps.sum(dim=(2,3)) > 0).float().unsqueeze(-1).unsqueeze(-1)
                loss = ((preds - heatmaps) ** 2) * valid_mask
                loss = loss.mean()

                val_loss += loss.item()

        val_loss /= len(test_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")

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

    print("\n========== LINE-WISE MEAN PIXEL ERROR ==========")
    total = []

    for name, errors in line_errors_dict.items():
        mean_err = np.mean(errors)
        total.extend(errors)
        print(f"{name}: {mean_err:.3f}")

    print("-----------------------------------------------")
    print(f"Overall Mean Line Pixel Error: {np.mean(total):.3f}")
    print("================================================\n")

    with open(PRED_JSON, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"✅ Predictions saved to {PRED_JSON}")


if __name__ == "__main__":
    train()
