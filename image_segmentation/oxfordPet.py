import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time


# U-Net 구현은 동일하므로 DoubleConv와 UNet 클래스는 이전과 같습니다
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()

        # Encoder
        self.conv1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        # Decoder
        up_1 = self.up1(c5)
        merge1 = torch.cat([up_1, c4], dim=1)
        c6 = self.conv6(merge1)
        up_2 = self.up2(c6)
        merge2 = torch.cat([up_2, c3], dim=1)
        c7 = self.conv7(merge2)
        up_3 = self.up3(c7)
        merge3 = torch.cat([up_3, c2], dim=1)
        c8 = self.conv8(merge3)
        up_4 = self.up4(c8)
        merge4 = torch.cat([up_4, c1], dim=1)
        c9 = self.conv9(merge4)

        output = self.conv10(c9)
        return torch.sigmoid(output)


# Oxford Pet 데이터셋을 위한 커스텀 데이터셋 클래스
class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split

        # 이미지와 마스크 경로 설정
        self.images_dir = self.root_dir / 'images'
        self.masks_dir = self.root_dir / 'annotations' / 'trimaps'

        # 데이터 분할 파일 읽기
        split_file = self.root_dir / 'annotations' / f'{split}.txt'
        with open(split_file, 'r') as f:
            self.image_list = [line.strip().split()[0] for line in f.readlines()]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]

        # 이미지 로드
        img_path = self.images_dir / f'{img_name}.jpg'
        mask_path = self.masks_dir / f'{img_name}.png'

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # 마스크 전처리 (trimap을 이진 마스크로 변환)
        mask = np.array(mask)
        mask = (mask == 2).astype(np.float32)  # 2는 pet 을 나타냄
        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # 학습 phase
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'    Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # 검증 phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Training Loss: {epoch_train_loss:.4f}')
        print(f'Validation Loss: {epoch_val_loss:.4f}')

        # 최고 성능 모델 저장
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
            }, 'best_model.pth')

    return train_losses, val_losses


def evaluate_model(model, test_loader, device, save_results=True):
    model.eval()
    test_images = []
    test_masks = []
    test_outputs = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            outputs = model(images)

            # CPU로 이동하고 numpy 배열로 변환
            test_images.extend(images.cpu())
            test_masks.extend(masks.cpu())
            test_outputs.extend(outputs.cpu())

            if len(test_images) >= 8:  # 8개 이미지만 저장
                break

    # 결과 시각화 및 저장
    fig, axes = plt.subplots(8, 3, figsize=(15, 30))
    plt.tight_layout()

    for idx in range(8):
        # 원본 이미지
        img = test_images[idx].permute(1, 2, 0)
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')

        # 실제 마스크
        axes[idx, 1].imshow(test_masks[idx].squeeze(), cmap='gray')
        axes[idx, 1].set_title('True Mask')
        axes[idx, 1].axis('off')

        # 예측 마스크
        pred_mask = (test_outputs[idx].squeeze() > 0.5).float()
        axes[idx, 2].imshow(pred_mask, cmap='gray')
        axes[idx, 2].set_title('Predicted Mask')
        axes[idx, 2].axis('off')

    if save_results:
        plt.savefig('test_results.png')
        plt.close()


def main():
    # 하이퍼파라미터 설정
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    IMAGE_SIZE = 256

    # 데이터셋 경로 설정
    dataset_root = 'oxford-iiit-pet'

    # 디바이스 설정
    device = get_device()
    print(f'Using device: {device}')

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # 데이터셋 준비
    full_dataset = OxfordPetDataset(root_dir=dataset_root, split='trainval', transform=transform)
    test_dataset = OxfordPetDataset(root_dir=dataset_root, split='test', transform=transform)

    # 학습 데이터를 학습/검증 세트로 분할
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    # 모델, 손실 함수, 옵티마이저 초기화
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 모델 학습
    print("Starting training...")
    start_time = time.time()
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion,
        optimizer, device, num_epochs=EPOCHS
    )
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")

    # 학습 곡선 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('learning_curves.png')
    plt.close()

    # 최고 성능 모델 로드
    print("Loading best model for testing...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 테스트 세트에서 모델 평가
    print("Evaluating model on test set...")
    evaluate_model(model, test_loader, device)
    print("Evaluation completed. Results saved as 'test_results.png'")


if __name__ == '__main__':
    main()