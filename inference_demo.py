import torch
import numpy as np
import random
import os
import cv2
import glob
import pandas as pd
from argparse import ArgumentParser
from scipy.ndimage import gaussian_filter

from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_test import cal_anomaly_map, min_max_norm, cvt2heatmap
from utils.utils_train import MultiProjectionLayer
from dataset.dataset import get_data_transforms


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_folder', default='./RD_Stain++', type=str)
    parser.add_argument('--checkpoint_name', default=None, type=str)

    parser.add_argument('--input_dir', required=True, type=str)
    parser.add_argument('--classes', nargs="+", default=["stain"])    
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--output_dir', default='./infer_output', type=str)
    parser.add_argument('--threshold', default=0.6, type=float)
    return parser.parse_args()


def load_model(checkpoint_folder, _class_, device, checkpoint_name=None):
    if checkpoint_name:
        checkpoint_path = os.path.join(checkpoint_folder, _class_, checkpoint_name)
    else:
        checkpoint_path = os.path.join(checkpoint_folder, _class_, f'wres50_{_class_}.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Cant find checkpoint: {checkpoint_path}")

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn      = bn.to(device)

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    proj_layer = MultiProjectionLayer(base=64).to(device)

    ckp = torch.load(checkpoint_path, map_location='cpu')
    proj_layer.load_state_dict(ckp['proj'])
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])

    encoder.eval()
    bn.eval()
    decoder.eval()
    proj_layer.eval()

    print(f"Loaded: {checkpoint_path}")
    return encoder, bn, decoder, proj_layer


def infer_one(img_path, encoder, bn, decoder, proj_layer,
              transform, image_size, threshold, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img / 255., (image_size, image_size))
    tensor = transform(img_resized).unsqueeze(0).float().to(device)

    with torch.no_grad():
        inputs   = encoder(tensor)
        features = proj_layer(inputs)
        outputs  = decoder(bn(features))

    anomaly_map, _ = cal_anomaly_map(inputs, outputs, tensor.shape[-1], amap_mode='a')
    print(anomaly_map)
    print("==========")
    anomaly_map     = gaussian_filter(anomaly_map, sigma=4)
    print(anomaly_map.max())
    print("==========min")
    print(anomaly_map.min())

    anomaly_score = float(np.max(anomaly_map))
    amap_norm     = min_max_norm(anomaly_map)
    binary_mask   = (amap_norm > threshold).astype(np.uint8) * 255

    return anomaly_score, amap_norm, binary_mask, img


# def visualize(img_orig, amap_norm, binary_mask, img_name, anomaly_score, image_size):
#     img_show = cv2.resize(img_orig, (image_size, image_size))
#     img_bgr  = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)

#     heatmap  = cvt2heatmap(amap_norm * 255)
#     heatmap  = cv2.resize(heatmap, (image_size, image_size))

#     mask_bgr = cv2.cvtColor(
#         cv2.resize(binary_mask, (image_size, image_size)), cv2.COLOR_GRAY2BGR)

#     heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     overlay  = np.float32(img_show) / 255 * 0.5 + np.float32(heatmap_rgb) / 255 * 0.5
#     overlay  = np.uint8(overlay * 255)
#     overlay  = cv2.resize(overlay, (image_size, image_size))
#     overlay  = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

#     for panel, title in zip([img_bgr, heatmap, mask_bgr, overlay],
#                              ["Input", "Anomaly Map", "Binary Mask", "Overlay"]):
#         cv2.putText(panel, title, (5, 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     combined = np.hstack([img_bgr, heatmap, mask_bgr, overlay])
#     header   = np.zeros((35, combined.shape[1], 3), dtype=np.uint8)
#     cv2.putText(header, f"{img_name}  |  Score: {anomaly_score:.4f}", (5, 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#     return np.vstack([header, combined])

def visualize(img_orig, binary_mask, img_name, anomaly_score, image_size):
    img_bgr = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)  # giữ size gốc
    h, w = img_bgr.shape[:2]
    
    mask_resized = cv2.resize(binary_mask, (w, h))  # resize mask về size gốc
    
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 2)
    
    cv2.putText(img_bgr, f"Score: {anomaly_score:.4f}", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_bgr

def run(_class_, pars):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    transform, _ = get_data_transforms(pars.image_size, pars.image_size)
    encoder, bn, decoder, proj_layer = load_model(
        pars.checkpoint_folder, _class_, device, pars.checkpoint_name
    )

    img_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        img_paths += glob.glob(os.path.join(pars.input_dir, ext))
        img_paths += glob.glob(os.path.join(pars.input_dir, '**', ext), recursive=True)
    img_paths = sorted(set(img_paths))

    if not img_paths:
        return

    out_dir = os.path.join(pars.output_dir, _class_)
    os.makedirs(out_dir, exist_ok=True)

    scores = []
    for i, img_path in enumerate(img_paths, 1):
        img_name = os.path.basename(img_path)

        anomaly_score, amap_norm, binary_mask, img_orig = infer_one(
            img_path, encoder, bn, decoder, proj_layer,
            transform, pars.image_size, pars.threshold, device
        )

        label = 'stain' if anomaly_score > pars.threshold else 'good'

        result = visualize(img_orig, amap_norm, binary_mask,
                           img_name, anomaly_score, pars.image_size)
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(img_name)[0] + '_result.jpg'), result)
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(img_name)[0] + '_mask.png'),
                    cv2.resize(binary_mask, (pars.image_size, pars.image_size)))

        scores.append({
            'image': img_name,
            'anomaly_score': round(anomaly_score, 6),
            'label': label
        })
        print(f"[{i}/{len(img_paths)}] {img_name} | score={anomaly_score:.4f} | {label.upper()}")

    df = pd.DataFrame(scores).sort_values('anomaly_score', ascending=False)
    df.to_csv(os.path.join(out_dir, 'scores.csv'), index=False)




if __name__ == '__main__':
    pars = get_args()
    setup_seed(111)
    for c in pars.classes:
        print(f"\n{'='*50}\nClass: {c}\n{'='*50}")
        run(c, pars)
