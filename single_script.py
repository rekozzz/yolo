# --- Setup Environment ---
!pip install -q ultralytics kagglehub
import kagglehub
import os
import shutil
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import glob
from tqdm import tqdm
import time

# --- Download and Extract Data ---
# Download latest version of Oxford-IIIT Pet dataset
path = kagglehub.dataset_download("tanlikesmath/the-oxfordiiit-pet-dataset")
print("Path to dataset files:", path)

dataset_dir = './dataset'
if not os.path.exists(dataset_dir):
    shutil.copytree(path, dataset_dir)
print("Data copied to", dataset_dir)

# --- Data Preprocessing (Train/Val/Test Split & YOLO format) ---
images_dir = os.path.join(dataset_dir, 'images/images')
annotations_dir = os.path.join(dataset_dir, 'annotations/annotations/xmls')
trimaps_dir = os.path.join(dataset_dir, 'annotations/annotations/trimaps')

# Get list of valid images that have annotations
all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
valid_samples = []

for img_name in tqdm(all_images, desc="Checking annotations"):
    base_name = os.path.splitext(img_name)[0]
    xml_path = os.path.join(annotations_dir, f"{base_name}.xml")
    trimap_path = os.path.join(trimaps_dir, f"{base_name}.png")
    if os.path.exists(xml_path) and os.path.exists(trimap_path):
        valid_samples.append(base_name)

print(f"Total valid samples: {len(valid_samples)}")

# Train/Val/Test Split (70/15/15)
train_samples, temp_samples = train_test_split(valid_samples, test_size=0.3, random_state=42)
val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)

print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

# Create YOLO directories
yolo_base_dir = './yolo_data'
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(yolo_base_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(yolo_base_dir, 'labels', split), exist_ok=True)

# Parse XML to YOLO
def convert_xml_to_yolo(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    bndbox = root.find('object').find('bndbox')
    xmin = float(bndbox.find('xmin').text)
    ymin = float(bndbox.find('ymin').text)
    xmax = float(bndbox.find('xmax').text)
    ymax = float(bndbox.find('ymax').text)
    
    x_center = ((xmin + xmax) / 2) / w
    y_center = ((ymin + ymax) / 2) / h
    width = (xmax - xmin) / w
    height = (ymax - ymin) / h
    
    return [0, x_center, y_center, width, height]  # Class 0: Pet

for split_name, split_list in zip(['train', 'val', 'test'], [train_samples, val_samples, test_samples]):
    for base_name in tqdm(split_list, desc=f"Processing {split_name} data"):
        xml_path = os.path.join(annotations_dir, f"{base_name}.xml")
        try:
            yolo_anno = convert_xml_to_yolo(xml_path)
            
            # Only copy image and create label if annotation parsing was successful
            src_img = os.path.join(images_dir, f"{base_name}.jpg")
            dst_img = os.path.join(yolo_base_dir, 'images', split_name, f"{base_name}.jpg")
            shutil.copy(src_img, dst_img)
            
            label_path = os.path.join(yolo_base_dir, 'labels', split_name, f"{base_name}.txt")
            with open(label_path, 'w') as f:
                f.write(f"{yolo_anno[0]} {yolo_anno[1]:.6f} {yolo_anno[2]:.6f} {yolo_anno[3]:.6f} {yolo_anno[4]:.6f}\n")
        except Exception as e:
            pass # Skip sample entirely to avoid empty labels

yaml_content = f"""
path: {os.path.abspath(yolo_base_dir)}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['pet']
"""
with open('dataset.yaml', 'w') as f:
    f.write(yaml_content)
print("YOLO dataset prepared.")

# --- YOLOv11 Training (Object Detection) ---
model = YOLO('yolo11n.pt')

results = model.train(
    data='dataset.yaml',
    epochs=10, 
    imgsz=640,
    batch=16,
    name='pet_yolo_model'
)

metrics = model.val()
print("\n--- YOLOv11 Evaluation Metrics ---")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}\n")

# --- U-Net Architecture Definition ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# --- U-Net Data Loading and Training ---
class PetDataset(Dataset):
    def __init__(self, sample_list, images_dir, trimaps_dir, transform=None):
        self.sample_list = sample_list
        self.images_dir = images_dir
        self.trimaps_dir = trimaps_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        base_name = self.sample_list[idx]
        img_path = os.path.join(self.images_dir, f"{base_name}.jpg")
        mask_path = os.path.join(self.trimaps_dir, f"{base_name}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        binary_mask = (mask == 1).astype(np.float32)

        if self.transform:
            image = self.transform(image)
            mask = transforms.functional.to_tensor(binary_mask)
            mask = transforms.functional.resize(mask, size=(128, 128), interpolation=transforms.InterpolationMode.NEAREST)

        return image, mask

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = PetDataset(train_samples, images_dir, trimaps_dir, transform)
val_dataset = PetDataset(val_samples, images_dir, trimaps_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_unet = UNet(in_channels=3, out_channels=1).to(device)
optimizer = optim.Adam(model_unet.parameters(), lr=1e-4)
bce_fn = nn.BCEWithLogitsLoss()

num_epochs = 15
train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training Phase
    model_unet.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    epoch_train_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)
        
        predictions = model_unet(data)
        loss = bce_fn(predictions, targets) + dice_loss(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation Phase
    model_unet.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        for data, targets in val_loop:
            data = data.to(device)
            targets = targets.to(device)
            
            predictions = model_unet(data)
            loss = bce_fn(predictions, targets) + dice_loss(predictions, targets)
            epoch_val_loss += loss.item()
            
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save Best Model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model_unet.state_dict(), 'best_unet.pth')
        print("--> Saved new best model!")

print("U-Net training complete.")

# --- Evaluation & Metrics ---
# Plot Training Curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (BCE + Dice)')
plt.title('U-Net Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Load best U-Net model
model_unet.load_state_dict(torch.load('best_unet.pth'))

test_dataset = PetDataset(test_samples, images_dir, trimaps_dir, transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

def compute_unet_metrics(loader, model, device):
    total_iou = 0
    total_dice = 0
    total_pixel_acc = 0
    total_precision = 0
    total_recall = 0
    
    model.eval()
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Evaluating U-Net"):
            data = data.to(device)
            targets = targets.to(device)
            
            preds = torch.sigmoid(model(data))
            preds = (preds > 0.5).float()
            
            # Flatten tensors
            preds_flat = preds.view(-1)
            targets_flat = targets.view(-1)
            
            intersection = (preds_flat * targets_flat).sum()
            union = preds_flat.sum() + targets_flat.sum() - intersection
            
            iou = (intersection + 1e-6) / (union + 1e-6)
            total_iou += iou.item()
            
            dice = (2. * intersection + 1e-6) / (preds_flat.sum() + targets_flat.sum() + 1e-6)
            total_dice += dice.item()
            
            pixel_acc = (preds_flat == targets_flat).float().mean()
            total_pixel_acc += pixel_acc.item()
            
            true_positives = intersection
            false_positives = (preds_flat * (1 - targets_flat)).sum()
            false_negatives = ((1 - preds_flat) * targets_flat).sum()
            
            precision = (true_positives + 1e-6) / (true_positives + false_positives + 1e-6)
            recall = (true_positives + 1e-6) / (true_positives + false_negatives + 1e-6)
            
            total_precision += precision.item()
            total_recall += recall.item()
            
    n = len(loader)
    print("\n--- U-Net Evaluation Metrics ---")
    print(f"Pixel Accuracy: {total_pixel_acc/n:.4f}")
    print(f"Precision:      {total_precision/n:.4f}")
    print(f"Recall:         {total_recall/n:.4f}")
    print(f"IoU:            {total_iou/n:.4f}")
    print(f"Dice Score:     {total_dice/n:.4f}\n")

compute_unet_metrics(test_loader, model_unet, device)

# --- Inference Speed Comparison ---
print("\n--- Inference Speed Comparison ---")
sample_img_path = os.path.join(images_dir, f"{test_samples[0]}.jpg")

# YOLO Inference Speed
start_yolo = time.time()
yolo_res = model.predict(sample_img_path, imgsz=640, verbose=False)
end_yolo = time.time()
yolo_time = (end_yolo - start_yolo) * 1000 # in ms

# U-Net Inference Speed
image = Image.open(sample_img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)
start_unet = time.time()
with torch.no_grad():
    _ = model_unet(input_tensor)
end_unet = time.time()
unet_time = (end_unet - start_unet) * 1000 # in ms

print(f"YOLOv11 Inference Time: {yolo_time:.2f} ms")
print(f"U-Net Inference Time:   {unet_time:.2f} ms\n")

# --- Side-by-Side Visual Comparison ---
def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

model_unet.eval()

num_samples = 3
fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

for i in range(num_samples):
    base_name = test_samples[i]
    img_path = os.path.join(images_dir, f"{base_name}.jpg")
    
    # YOLO Inference
    yolo_res = model.predict(img_path, imgsz=640, verbose=False)
    yolo_img = yolo_res[0].plot()
    
    # U-Net Inference
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model_unet(input_tensor))
        preds = (preds > 0.5).float()
    
    orig_img = unnormalize(input_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()
    orig_img = np.clip(orig_img, 0, 1)
    mask_img = preds.squeeze(0).squeeze(0).cpu().numpy()
    
    axes[i, 0].imshow(np.array(Image.open(img_path)))
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(yolo_img[..., ::-1])
    axes[i, 1].set_title("YOLOv11 Detection")
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(orig_img)
    axes[i, 2].imshow(mask_img, alpha=0.5, cmap='jet')
    axes[i, 2].set_title("U-Net Segmentation Mask")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()
