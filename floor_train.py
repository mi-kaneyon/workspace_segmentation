import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# U-Netの定義 (Batch Normを追加したバージョン)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        # Final output layer
        self.final = nn.Conv2d(64, 2, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(kernel_size=2)(enc1))
        bottleneck = self.bottleneck(nn.MaxPool2d(kernel_size=2)(enc2))
        
        dec2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        
        return self.final(dec1)

# データ増強の設定
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])

# データセットクラス
class FloorSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)

        # マスクファイル名の生成
        if '_mask' in image_filename:
            mask_filename = image_filename
        else:
            mask_filename = image_filename.replace('.png', '_mask.png')

        mask_path = os.path.join(self.mask_dir, mask_filename)

        # 画像とマスクの読み込み
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 変換の適用
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)

        # マスクの255を1に変換し、データ型をtorch.LongTensorに変更
        mask = mask.squeeze(0)  # チャンネルの次元を削除
        mask[mask == 255] = 1
        mask = mask.long()  # LongTensorに変換

        return image, mask



# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# データセットの読み込みとデータローダの設定
dataset = FloorSegmentationDataset(image_dir='corridor/BlendedData', mask_dir='corridor/BlendedData', transform=data_transforms)
train_set, val_set = train_test_split(dataset, test_size=0.2)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

# モデルの初期化
model = UNet().to(device)

# 損失関数と最適化関数の設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
num_epochs = 20
best_val_loss = float('inf')
early_stop_count = 0
early_stop_limit = 5
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (loop.n + 1))
    
    train_losses.append(running_loss / len(train_loader))
    
    # 検証ループ
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1} finished, Train Loss: {train_losses[-1]}, Val Loss: {val_loss}")
    
    # Early Stoppingのチェック
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        early_stop_count = 0
    else:
        early_stop_count += 1
    
    if early_stop_count >= early_stop_limit:
        print("Early stopping triggered.")
        break

# 学習履歴のプロット
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.show()

print("最良モデルを保存しました: best_model.pth")
