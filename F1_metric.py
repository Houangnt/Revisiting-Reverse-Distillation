import torch
import numpy as np
import random
import os
import cv2
import glob
import pandas as pd

from argparse import ArgumentParser
from scipy.ndimage import gaussian_filter
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve

from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_test import cal_anomaly_map, min_max_norm
from utils.utils_train import MultiProjectionLayer
from dataset.dataset import get_data_transforms
import time

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_folder', default='./RD_Stain++', type=str)
    parser.add_argument('--input_dir', required=True, type=str)   
    parser.add_argument('--gt_dir', required=True, type=str)      
    parser.add_argument('--class_name', default='stain', type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--output_dir', default='./infer_output_v2', type=str)
    return parser.parse_args()


def load_model(checkpoint_folder, class_name, device):
    checkpoint_path = os.path.join(checkpoint_folder, class_name, f'wres50_{class_name}.pth')

    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)
    proj = MultiProjectionLayer(base=64)

    ckp = torch.load(checkpoint_path, map_location='cpu')
    proj.load_state_dict(ckp['proj'])
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])

    encoder, bn, decoder, proj = encoder.to(device), bn.to(device), decoder.to(device), proj.to(device)

    encoder.eval(); bn.eval(); decoder.eval(); proj.eval()

    print(f"Loaded: {checkpoint_path}")
    return encoder, bn, decoder, proj


def infer_one(img_path, encoder, bn, decoder, proj, transform, image_size, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start = time.time()

    img_resized = cv2.resize(img / 255., (image_size, image_size))
    tensor = transform(img_resized).unsqueeze(0).float().to(device)
    

    with torch.no_grad():
        inputs = encoder(tensor)
        features = proj(inputs)
        outputs = decoder(bn(features))

    anomaly_map, _ = cal_anomaly_map(inputs, outputs, image_size, amap_mode='a')
    anomaly_map = gaussian_filter(anomaly_map, sigma=4)

    anomaly_score = float(np.max(anomaly_map))
    end = time.time()
    print("TIME ", end-start)

    return anomaly_map, anomaly_score 


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform, _ = get_data_transforms(args.image_size, args.image_size)

    encoder, bn, decoder, proj = load_model(args.checkpoint_folder, args.class_name, device)

    img_paths = []
    labels = []

    for cls in ['good', args.class_name]:
        cls_dir = os.path.join(args.input_dir, cls)
        paths = sorted(glob.glob(os.path.join(cls_dir, '*.*')))
        img_paths.extend(paths)
        labels.extend([0 if cls == 'good' else 1] * len(paths))

    all_gt_px = []
    all_pred_px = []
    all_scores_img = []
    all_labels_img = []

    os.makedirs(args.output_dir, exist_ok=True)

    for i, (img_path, img_label) in enumerate(zip(img_paths, labels), 1):
        name = os.path.basename(img_path)

        anomaly_map, anomaly_score = infer_one(
            img_path, encoder, bn, decoder, proj,
            transform, args.image_size, device
        )

        if img_label == 0:
            gt = np.zeros_like(anomaly_map, dtype=np.uint8)
        else:
            gt_name = os.path.splitext(name)[0] + ".png"
            gt_path = os.path.join(args.gt_dir, gt_name)

            if not os.path.exists(gt_path):
                print(f"Missing GT: {name}")
                continue

            gt = cv2.imread(gt_path, 0)
            gt = cv2.resize(gt, (anomaly_map.shape[1], anomaly_map.shape[0]))
            gt = (gt > 127).astype(np.uint8)

        all_gt_px.extend(gt.ravel())
        all_pred_px.extend(anomaly_map.ravel())
        all_scores_img.append(anomaly_score)
        all_labels_img.append(img_label)

        print(f"[{i}/{len(img_paths)}] {name} | score={anomaly_score:.4f}")

    all_gt_px      = np.array(all_gt_px)
    all_pred_px    = np.array(all_pred_px)
    all_scores_img = np.array(all_scores_img)
    all_labels_img = np.array(all_labels_img)

    # ── Pixel-level metrics (giữ nguyên) ──────────────────────────
    px_precision, px_recall, px_thresholds = precision_recall_curve(all_gt_px, all_pred_px)
    px_f1_scores = np.divide(
        2 * px_precision * px_recall,
        px_precision + px_recall,
        out=np.zeros_like(px_precision),
        where=(px_precision + px_recall) != 0,
    )
    best_px_idx = np.argmax(px_f1_scores[:-1])
    best_px_th  = px_thresholds[best_px_idx]
    pred_px_bin = (all_pred_px >= best_px_th).astype(np.uint8)

    px_f1        = f1_score(all_gt_px, pred_px_bin)
    px_precision_val = precision_score(all_gt_px, pred_px_bin)
    px_recall_val    = recall_score(all_gt_px, pred_px_bin)
    auroc_px     = roc_auc_score(all_gt_px, all_pred_px)

    img_precision, img_recall, img_thresholds = precision_recall_curve(all_labels_img, all_scores_img)
    img_f1_scores = np.divide(
        2 * img_precision * img_recall,
        img_precision + img_recall,
        out=np.zeros_like(img_precision),
        where=(img_precision + img_recall) != 0,
    )
    best_img_idx = np.argmax(img_f1_scores[:-1])
    best_img_th  = img_thresholds[best_img_idx]
    pred_img_bin = (all_scores_img >= best_img_th).astype(np.uint8)

    img_f1           = f1_score(all_labels_img, pred_img_bin)
    img_precision_val = precision_score(all_labels_img, pred_img_bin)
    img_recall_val    = recall_score(all_labels_img, pred_img_bin)
    auroc_img        = roc_auc_score(all_labels_img, all_scores_img)

    print("\n" + "="*50)
    print("[Pixel-level]")
    print(f"  Best threshold : {best_px_th:.6f}")
    print(f"  F1-score       : {px_f1:.4f}")
    print(f"  Precision      : {px_precision_val:.4f}")
    print(f"  Recall         : {px_recall_val:.4f}")
    print(f"  AUROC          : {auroc_px:.4f}")

    print("[Image-level]")
    print(f"  Best threshold : {best_img_th:.6f}")
    print(f"  F1-score       : {img_f1:.4f}")
    print(f"  Precision      : {img_precision_val:.4f}")
    print(f"  Recall         : {img_recall_val:.4f}")
    print(f"  AUROC          : {auroc_img:.4f}")
    print("="*50)


if __name__ == '__main__':
    args = get_args()
    setup_seed(111)
    run(args)
