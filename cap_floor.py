import torch
import torch.nn as nn
import cv2
import numpy as np

# U-Netの定義（BatchNormを含む）
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 最終出力レイヤー
        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)  # size: [batch, 64, H/2, W/2]
        enc2 = self.enc2(enc1)  # size: [batch, 128, H/4, W/4]
        bottleneck = self.bottleneck(enc2)  # size: [batch, 256, H/4, W/4]

        # Decoder
        dec2 = self.up2(bottleneck)  # size: [batch, 128, H/2, W/2]
        # enc2のサイズをdec2に合わせる
        enc2_resized = nn.functional.interpolate(enc2, size=dec2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat([dec2, enc2_resized], dim=1))  # size: [batch, 128, H/2, W/2]

        dec1 = self.up1(dec2)  # size: [batch, 64, H, W]
        # enc1のサイズをdec1に合わせる
        enc1_resized = nn.functional.interpolate(enc1, size=dec1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat([dec1, enc1_resized], dim=1))  # size: [batch, 64, H, W]

        return self.final(dec1)  # size: [batch, 2, H, W]

# デバイスの設定 (GPUまたはCPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# モデルの初期化と学習済みモデルのロード
model = UNet().to(device)
model.load_state_dict(torch.load('best_model.pth'))  # 学習済みの最良モデルをロード
model.eval()  # モデルを評価モードに設定

# カメラのキャプチャを開始
cap = cv2.VideoCapture(2)  # 0はデフォルトのカメラデバイスを指します

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # フレームの前処理
    input_frame = cv2.resize(frame, (640, 480))  # フレームのリサイズ
    input_tensor = torch.from_numpy(input_frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    # 推論の実行
    with torch.no_grad():
        output = model(input_tensor)
        predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # マスクをカラー化
    mask_colored = np.zeros_like(frame)
    mask_colored[predicted_mask == 1] = [0, 255, 0]  # フロア領域を緑色に設定

    # マスクを元のフレームに重ね合わせる
    alpha = 0.5  # マスクの透明度
    overlay_frame = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)

    # 結果を表示
    cv2.imshow('Floor Segmentation', overlay_frame)

    # 'q'キーを押すとループ終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラの解放とウィンドウの閉鎖
cap.release()
cv2.destroyAllWindows()
