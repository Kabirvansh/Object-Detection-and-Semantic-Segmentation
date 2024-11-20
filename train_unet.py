import numpy as np
import torch
from torch.utils.data import Dataset

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
# Dataset class to load images and masks from .npz files
class NPZImageMaskDataset(Dataset):
    def __init__(self, npz_path):
        self.data = np.load(npz_path)
        self.images = self.data['images']  # Assuming images are stored in 'images'
        self.masks = self.data['semantic_masks']  # Assuming masks are stored in 'semantic_masks'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Reshape images and normalize
        image = self.images[idx].reshape(64, 64, 3).astype(np.float32) / 255.0  # Reshape to 64x64x3 and normalize
        mask = self.masks[idx].reshape(64, 64).astype(np.int64)  # Reshape mask to 64x64

        # Convert to tensor
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # Shape: [3, 64, 64]
        mask = torch.tensor(mask, dtype=torch.long)  # Shape: [64, 64]

        return image, mask
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=11):
        super().__init__()
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)
        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

def compute_val_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # Get the predicted class for each pixel
            correct += (preds == masks).sum().item()  # Count correctly classified pixels
            total_pixels += masks.numel()

    return correct / total_pixels if total_pixels > 0 else 0

# Training function
def train_unet(train_npz_path, val_npz_path, num_epochs=10, batch_size=32, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training and validation data from .npz files
    train_dataset = NPZImageMaskDataset(train_npz_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataset = NPZImageMaskDataset(val_npz_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model, loss function, and optimizer
    unet_model = UNET(in_channels=3, out_channels=11).to(device)
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class segmentation
    optimizer = optim.Adam(unet_model.parameters(), lr=learning_rate)

    # Directory for saving models on Colab
    save_dir = "/content/models"
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        unet_model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = unet_model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Calculate average loss for this epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Compute validation accuracy
        val_acc = compute_val_accuracy(unet_model, val_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_acc:.4f}")

        # Save the model with epoch and validation accuracy in the filename
        model_path = os.path.join(save_dir, f"unet_epoch_{epoch+1}_val_acc_{val_acc:.4f}.pth")
        torch.save(unet_model.state_dict(), model_path)

    print("Training complete.")

if __name__ == "__main__":
    # Set your .npz file paths for Colab
    train_npz_path = "/content/train.npz"
    val_npz_path = "/content/valid.npz"

    # Run the training
    train_unet(train_npz_path, val_npz_path, num_epochs=10, batch_size=32, learning_rate=1e-4)