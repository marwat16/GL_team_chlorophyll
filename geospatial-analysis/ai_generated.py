# TODO:  1. Feel free to pip install and import any useful libraries within the
#  rewrite cell.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp import GradScaler, autocast
# Constants for model training
EPOCHS = 225
NUM_CLASSES = 17
IMG_SIZE = 256 # DLRSD images are 256x256
BATCH_SIZE = 16 # Adjust based on GPU memory
LEARNING_RATE = 1e-4
# Device configuration (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset Class
class DLRSDDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform
    def __len__(self):
        return len(self.data_paths)
    def __getitem__(self, idx):
        image_path, label_path = self.data_paths[idx]
        # Load image and ensure 3 channels (RGB)
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # Load label mask. Labels are 1-indexed (1 to 17) in the dataset.
        # PyTorch's CrossEntropyLoss expects 0-indexed targets (0 to 16).
        label_mask = np.array(Image.open(label_path), dtype=np.int64) - 1
        if self.transform:
            # Albumentations expects images as HWC and masks as HW
            transformed = self.transform(image=image, mask=label_mask)
            image = transformed["image"]
            label_mask = transformed["mask"]
        return image, label_mask
# Define transformation pipelines for training and validation
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.85), # MODIFICATION: Increased flip probability from 0.75 to 0.85
    A.VerticalFlip(p=0.85),   # MODIFICATION: Increased flip probability from 0.75 to 0.85
 
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=20, p=0.5), # MODIFICATION: Reverted rotate_limit from 15 to 20
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5), # MODIFICATION: Increased brightness and contrast limits
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
# Main Model Wrapper Class
class SemanticSegmentationModel:
    def __init__(self, num_classes=NUM_CLASSES):
        # Initialize U-Net model with a pre-trained ResNeXt101 encoder (MODIFICATION: Upgraded encoder)
        self.model = smp.Unet(
            encoder_name="se_resnext101_32x4d", # MODIFICATION
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            decoder_dropout=0.05, # MODIFICATION: Adjusted decoder_dropout for regularization (from 0.1 to 0.05)
        )
        self.model.to(device)
        # Switched to AdamW optimizer for better performance with weight decay
        # MODIFICATION: Reverted weight_decay from 5e-5 to 1e-4
        # NEW MODIFICATION: Adjusted weight_decay from 1e-4 to 7.5e-5
        # CURRENT MODIFICATION: Reverted weight_decay from 7.5e-5 to 1e-4
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # MODIFIED
        
        # Define combined loss function: 60% Dice Loss + 40% Cross-Entropy Loss (MODIFICATION)
        dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        ce_loss = nn.CrossEntropyLoss()
        
        # Wrapper function for the combined loss
        def combined_loss(y_pred, y_true):
            return 0.6 * dice_loss(y_pred, y_true) + 0.4 * ce_loss(y_pred, y_true) # MODIFICATION
        
        self.criterion = combined_loss
        
        # Initialize learning rate scheduler, added min_lr
        # MODIFICATION: Decreased patience from 10 to 7
        # PREVIOUS MODIFICATION: Changed factor from 0.5 to 0.7 for less aggressive decay
        # CURRENT MODIFICATION: Reverted factor from 0.7 to 0.5 for more aggressive decay
        # NEW MODIFICATION: Reverted patience from 7 to 10
        # NEW MODIFICATION: Changed factor from 0.7 to 0.5 for more aggressive decay
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7) # MODIFIED
        
        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler('cuda')
    def train_model(self, training_data_paths, epochs=EPOCHS):
        """
        Trains the segmentation model using the provided training data paths.
        """
        train_dataset = DLRSDDataset(training_data_paths, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # MODIFICATION: Adjusted num_workers from 8 to 4
        print(f"Starting training for {epochs} epochs on {device}...")
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device)
                # Ensure masks are of type torch.long for CrossEntropyLoss
                masks = masks.long() 
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with autocast('cuda'):
                    outputs = self.model(images)
                    # Calculate loss
                    loss = self.criterion(outputs, masks)
                
                # Backward pass and optimization with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            self.scheduler.step(avg_loss)
        print("Training complete.")
    def segment_single_image(self, image: np.ndarray) -> np.ndarray:
        """
        Predicts the segmentation mask for a single input image using Test-Time Augmentation (TTA).
        Args:
            image: A NumPy array representing the input image (H, W, C).
        Returns:
            A NumPy array representing the predicted segmentation mask (H, W),
            with pixel values corresponding to 1-indexed class IDs (1 to 17).
        """
        self.model.eval()
        with torch.no_grad():
            # List to store predictions (logits) from different augmentations
            all_logits = []
            # Prepare original image tensor
            transformed_original = val_transform(image=image)
            image_tensor_original = transformed_original["image"].unsqueeze(0).to(device) # Shape: (1, C, H, W)
            # 1. Original image
            with autocast('cuda'):
                logits_original = self.model(image_tensor_original)
            all_logits.append(logits_original)
            # 2. Horizontal flip
            # Apply flip on image (NumPy) then re-transform
            image_hflip = A.HorizontalFlip(p=1.0)(image=image)["image"]
            transformed_hflip = val_transform(image=image_hflip)
            image_tensor_hflip = transformed_hflip["image"].unsqueeze(0).to(device)
            with autocast('cuda'):
                logits_hflip = self.model(image_tensor_hflip)
            # Reverse horizontal flip on logits (flip along width dimension, which is 3)
            all_logits.append(torch.flip(logits_hflip, dims=[3]))
            # 3. Vertical flip
            # Apply flip on image (NumPy) then re-transform
            image_vflip = A.VerticalFlip(p=1.0)(image=image)["image"]
            transformed_vflip = val_transform(image=image_vflip)
            image_tensor_vflip = transformed_vflip["image"].unsqueeze(0).to(device)
            with autocast('cuda'):
                logits_vflip = self.model(image_tensor_vflip)
            # Reverse vertical flip on logits (flip along height dimension, which is 2)
            all_logits.append(torch.flip(logits_vflip, dims=[2]))
            # 4. Horizontal + Vertical flip
            # Apply both flips on image (NumPy) then re-transform
            image_hvflip = A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)])(image=image)["image"]
            transformed_hvflip = val_transform(image=image_hvflip)
            image_tensor_hvflip = transformed_hvflip["image"].unsqueeze(0).to(device)
            with autocast('cuda'):
                logits_hvflip = self.model(image_tensor_hvflip)
            # Reverse both flips on logits
            all_logits.append(torch.flip(torch.flip(logits_hvflip, dims=[3]), dims=[2]))
            # --- NEW ADDITIONS: Rotations (k=1 for 90, k=2 for 180, k=3 for 270 degrees clockwise) ---
            # For these, it's easier to apply torch.rot90 directly on the tensor
            # 5. Rotate 90 degrees clockwise
            image_tensor_rot90 = torch.rot90(image_tensor_original, k=1, dims=[2, 3])
            with autocast('cuda'):
                logits_rot90 = self.model(image_tensor_rot90)
            all_logits.append(torch.rot90(logits_rot90, k=-1, dims=[2, 3])) # Revert 90 deg counter-clockwise
            # 6. Rotate 180 degrees
            image_tensor_rot180 = torch.rot90(image_tensor_original, k=2, dims=[2, 3])
            with autocast('cuda'):
                logits_rot180 = self.model(image_tensor_rot180)
            all_logits.append(torch.rot90(logits_rot180, k=-2, dims=[2, 3])) # Revert 180 deg
            # 7. Rotate 270 degrees clockwise
            image_tensor_rot270 = torch.rot90(image_tensor_original, k=3, dims=[2, 3])
            with autocast('cuda'):
                logits_rot270 = self.model(image_tensor_rot270)
            all_logits.append(torch.rot90(logits_rot270, k=-3, dims=[2, 3])) # Revert 270 deg counter-clockwise
            # Average the logits from all augmented predictions
            averaged_logits = torch.mean(torch.stack(all_logits), dim=0)
            # Get class with highest probability for each pixel
            predicted_mask = torch.argmax(averaged_logits, dim=1).squeeze(0)
            # Convert to NumPy array and adjust to 1-indexed class IDs (0-indexed to 1-indexed)
            pred_mask_np = predicted_mask.cpu().numpy() + 1
        return pred_mask_np
    
    # model = SemanticSegmentationModel()
    # model.train_model(training_data_paths)

if __name__ == "__main__":
    # only needed for sub testing
    import argparse
    import json
    import random
    from pathlib import Path

    seed = 42
    this_folder = Path(__file__).resolve().parent
    train_data_file = this_folder / "train_split.json"
    test_data_file = this_folder / "test_split.json"
    total_iou = 0
    num_of_classes = 0

    # only needed for sub testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--subset", type=int, default=0)
    args = parser.parse_args()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load split data:
    with open(train_data_file) as f:
        train_data_pairs = [(this_folder / image_path, this_folder / label_path)
                       for image_path, label_path in json.load(f)]
    with open(test_data_file) as f:
        test_data_pairs = [(this_folder / image_path, this_folder / label_path)
                      for image_path, label_path in json.load(f)]

    # only needed for sub testing
    if args.subset:
        rng = random.Random(0)
        train_data_pairs = rng.sample(train_data_pairs, args.subset)
        test_data_pairs = rng.sample(test_data_pairs, args.subset)

    # training:
    model = SemanticSegmentationModel()
    model.train_model(train_data_pairs, epochs=args.epochs)

    # evaluate: 
    intersection = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)

    for image_path, label_path in test_data_pairs:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = np.array(image)

        label = Image.open(label_path)
        label = np.array(label)

        prediction = model.segment_single_image(image)

        for class_number in range(1, NUM_CLASSES + 1):
            predicted_pixels = prediction == class_number
            real_pixels = label == class_number

            same_pixels = predicted_pixels & real_pixels
            all_pixels = predicted_pixels | real_pixels

            intersection[class_number - 1] += same_pixels.sum()
            union[class_number - 1] += all_pixels.sum()

    for i in range(NUM_CLASSES):
        if union[i] > 0:
            iou = intersection[i] / union[i]
            total_iou += iou
            num_of_classes += 1

    miou = total_iou / num_of_classes
    print("mIoU:", miou)
