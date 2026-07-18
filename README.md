# YOLO vs U-Net on the Oxford-IIIT Pet Dataset

This project compares two computer vision approaches on the Oxford-IIIT Pet Dataset:

- **YOLOv11** for **object detection** using pet bounding boxes
- **U-Net** for **semantic segmentation** using pixel-level trimap masks

The repository includes both a Jupyter notebook and a Python script that download the dataset, prepare training data, train the models, evaluate them, and compare inference speed and visual outputs.

## Overview

The project is designed to answer a simple question: **how do object detection and semantic segmentation compare on the same dataset?**

- YOLOv11 predicts a bounding box around each pet.
- U-Net predicts a binary mask that outlines the pet more precisely.

The notebook and script also include:

- dataset download and extraction
- train/validation/test splitting
- conversion of Oxford-IIIT annotations into YOLO format
- U-Net training with BCE + Dice loss
- evaluation metrics for both models
- side-by-side visual comparison

## Dataset

The project uses the **Oxford-IIIT Pet Dataset**, which contains:

- pet images
- XML bounding box annotations
- trimap masks for segmentation

The preprocessing pipeline:

1. Downloads the images and annotations archives
2. Finds valid image/annotation/mask triplets
3. Splits the data into **70% train / 15% validation / 15% test**
4. Converts bounding box XML annotations into YOLO label format
5. Converts trimaps into binary segmentation masks for U-Net

## Models

### YOLOv11

Used for object detection.

- Initialized from `yolo11n.pt`
- Trains on the prepared YOLO dataset
- Reports precision, recall, mAP50, and mAP50-95
- Produces bounding boxes for pet localization

### U-Net

Used for semantic segmentation.

- Implemented from scratch in PyTorch
- Uses encoder-decoder architecture with skip connections
- Trained with **BCEWithLogitsLoss + Dice loss**
- Evaluated with pixel accuracy, precision, recall, IoU, and Dice score

## Repository Contents

- `Oxford_Pet_YOLO_UNet.ipynb` — main notebook version of the project
- `single_script.py` — a single Python script version of the full workflow
- `generate_notebook.py` — script used to generate the notebook
- `Project_Report.md` — written report describing the project and results

## Requirements

The project uses Python and common deep learning libraries, including:

- `torch`
- `torchvision`
- `ultralytics`
- `numpy`
- `matplotlib`
- `Pillow`
- `scikit-learn`
- `tqdm`
- `opencv-python`
- `kagglehub` (imported in the scripts, though the dataset download in the current workflow uses direct URLs)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rekozzz/yolo.git
cd yolo
```

### 2. Install dependencies

```bash
pip install torch torchvision ultralytics numpy matplotlib pillow scikit-learn tqdm opencv-python kagglehub
```

### 3. Run the notebook

Open `Oxford_Pet_YOLO_UNet.ipynb` in Jupyter or Google Colab and run the cells from top to bottom.

### 4. Or run the Python script

```bash
python single_script.py
```

## Workflow

The main workflow is:

1. Download Oxford-IIIT Pet data
2. Prepare YOLO labels and segmentation masks
3. Train YOLOv11 for detection
4. Train U-Net for segmentation
5. Evaluate both models
6. Compare inference speed
7. Visualize predictions side by side

## Outputs

The project generates artifacts such as:

- `dataset.yaml` — YOLO dataset configuration
- `yolo_data/` — prepared YOLO-format dataset
- `runs/detect/pet_yolo_model/` — YOLO training outputs
- `best_unet.pth` — best saved U-Net weights
- training curves and comparison plots

## Notes

- The notebook is designed to run in **Google Colab**.
- The YOLO training configuration currently assumes a CUDA-capable device with `device=0`.
- The project report includes spaces for final metric values and figures that can be filled in after training.

## Results Summary

The report and notebook suggest that:

- **YOLOv11** is better suited for fast coarse localization
- **U-Net** is better suited for precise pixel-level segmentation

This makes the repository a useful comparison of detection versus segmentation on a shared dataset.

## License

No license file is currently included in the repository.
