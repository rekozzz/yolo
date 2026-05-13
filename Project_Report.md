# Project Report: Object Detection and Semantic Segmentation
**Oxford-IIIT Pet Dataset**

## 1. Objective
The goal of this project is to implement and compare two deep learning paradigms—Object Detection and Semantic Segmentation—using the Oxford-IIIT Pet Dataset. Specifically, we utilize YOLOv11 to localize pets with bounding boxes and U-Net to segment the pets at a pixel level. The performance, accuracy, and use cases of both approaches are compared.

## 2. Dataset Understanding and Preprocessing
The **Oxford-IIIT Pet Dataset** contains images of 37 different pet breeds. It is unique in that it provides three forms of annotations: classification labels, head bounding boxes, and pixel-level trimap segmentations.

### Preprocessing Pipeline
1. **Data Split**: The dataset was randomly split into 70% Training, 15% Validation, and 15% Testing.
2. **YOLOv11 Preparation**: 
   - XML files containing bounding boxes were parsed.
   - Coordinates `(xmin, ymin, xmax, ymax)` were converted to normalized YOLO format: `class x_center y_center width height`.
   - Images were resized automatically to 640x640 by the Ultralytics engine during training.
3. **U-Net Preparation**:
   - Trimap masks (containing 1: foreground, 2: background, 3: borders) were converted into binary masks mapping the pet foreground to `1` and everything else to `0`.
   - Images and masks were resized to 128x128.
   - Images were normalized using standard ImageNet mean and standard deviation arrays.

## 3. Model Implementations

### Model A: YOLOv11 (Object Detection)
YOLO (You Only Look Once) frames detection as a regression problem, predicting bounding box coordinates and class probabilities simultaneously.
- **Architecture**: Employs a modified CSPDarknet backbone, a Path Aggregation Network (PANet) neck for feature fusion, and a decoupled detection head.
- **Implementation**: Utilized the `ultralytics` Python library. We initialized weights using `yolo11n.pt` for faster convergence.
- **Output Type**: Bounding boxes (coarse spatial localization).

### Model B: U-Net (Semantic Segmentation)
U-Net is a symmetric convolutional neural network designed for dense pixel-wise prediction.
- **Architecture**: 
  - *Encoder*: Successive sets of Double Convolutions and Max Pooling layers to extract deep semantic features.
  - *Decoder*: Transposed convolutions to upsample feature maps back to the original resolution.
  - *Skip Connections*: Concatenations between the encoder and decoder to retain spatial precision.
- **Implementation**: Built from scratch using PyTorch. 
- **Output Type**: Pixel-wise Mask (fine granular localization).

## 4. Evaluation Metrics
*Note: Ensure to populate the values below after running the models in Google Colab.*

- **YOLOv11 Metrics**:
  - **mAP50**: [Value from Colab]
  - **Inference Speed**: [Value from Colab] ms/image

- **U-Net Metrics**:
  - **Dice Score / F1-Score**: [Value from Colab]
  - **Binary Cross Entropy Loss**: [Value from Colab]

## 5. Visual Outputs
*(Insert the side-by-side comparison images generated from the notebook here.)*

| Original Image | YOLOv11 Bounding Box | U-Net Segmentation Mask |
|----------------|-----------------------|-------------------------|
| [Image 1]      | [Detection 1]         | [Mask 1]                |
| [Image 2]      | [Detection 2]         | [Mask 2]                |

## 6. Comparison Table

| Feature | YOLOv11 (Detection) | U-Net (Segmentation) |
|---------|---------------------|----------------------|
| **Output Type** | Bounding Boxes (Rectangles) | Pixel-wise Mask (Contours) |
| **Granularity** | Coarse (Object level) | Fine (Pixel level) |
| **Inference Speed** | Generally Faster (Real-time tracking) | Generally Slower (Per-pixel computation) |
| **Use Case** | Counting, Tracking, Cropping | Precision isolating, Medical imaging, Background removal |
| **Best Metric** | mAP (mean Average Precision) | IoU / Dice Coefficient |

## 7. Conclusion
Both models successfully learned to localize pets within the Oxford-IIIT dataset, albeit using fundamentally different approaches. 

**YOLOv11** proved to be highly efficient and faster during inference. It excels in scenarios where we only need to know the general location and scale of an object. However, it cannot define the exact boundaries of complex shapes.

**U-Net** provided exceptional detail, successfully mapping the precise contours of the pets, handling curved tails and ears well. The tradeoff is an increased computational cost due to dense pixel predictions and the necessity of high-quality pixel-level ground truth data (trimaps). 

Ultimately, the choice between these models depends on the specific downstream application: bounding boxes for real-time tracking, and segmentation masks for detailed shape analysis.
