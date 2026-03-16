# RD++ — Revisiting Reverse Distillation for Anomaly Detection

> **CVPR 2023** · [📄 Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf) · [🎥 Video](https://www.youtube.com/watch?v=cGRgy2Z0XQo&t=37s) · [🚀 Colab](https://colab.research.google.com/github/tientrandinh/Revisiting-Reverse-Distillation/blob/main/main.ipynb)

![Method Overview](./docs/method_training.png)

---

## Overview

**RD++** improves anomaly detection through a multi-task learning design with two key contributions:

- **Feature Compactness** — Self-supervised optimal transport to tighten normal feature representations.
- **Anomalous Signal Suppression** — Simplex-noise-based pseudo-abnormal samples to minimize reconstruction of anomalous patterns.

RD++ achieves **state-of-the-art** on MVTec for both anomaly detection and localization, while being:
- **6× faster** than PatchCore
- **2× faster** than CFA
- Negligible overhead over the original RD

![Inference Time](./docs/inference_time.jpeg)

---

## Quick Start

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

Organize your custom dataset as follows:

```
Stain_Dataset/
└── stain/
    ├── train/
    │   └── good/
    │       ├── 001_good.jpg
    │       └── ...
    ├── test/
    │   ├── good/
    │   │   └── ...
    │   └── stain/
    │       └── ...
    └── ground_truth/
        └── stain/
            └── ...
```

---

## Train

```bash
CUDA_VISIBLE_DEVICES=2,3 python main.py \
  --save_folder RD_Stain++ \
  --classes stain \
  --batch_size 32 \
  --image_size 256 \
  --data_path Stain_Dataset \
  --num_epoch 50
```

---

## Demo

```bash
python inference_demo.py
  --checkpoint_name wres50_stain_ep100.pth \
  --classes stain \
  --input_dir Good/ \
  --threshold 0.5
```

---

## Citation

```bibtex
@InProceedings{Tien_2023_CVPR,
    author    = {Tien, Tran Dinh and Nguyen, Anh Tuan and Tran, Nguyen Hoang and
                 Huy, Ta Duc and Duong, Soan T.M. and Nguyen, Chanh D. Tr. and
                 Truong, Steven Q. H.},
    title     = {Revisiting Reverse Distillation for Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {24511-24520}
}
```

---

## Contact & Acknowledgements

Questions? Open an issue or email **trandinhtienftu95@gmail.com**.

Built on top of [RD4AD](https://github.com/hq-deng/RD4AD) and [AnoDDPM (Simplex Noise)](https://github.com/Julian-Wyatt/AnoDDPM). Thanks to their brilliant work!
