import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from model import MCCDSNet
from torchvision.transforms import InterpolationMode
from bayes_opt import BayesianOptimization
import pandas as pd

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

class_names = ['Background', 'Fracture', 'Inorganic minerals', 'Inorganic pore',
               'Organic matter', 'Organic pores', 'Pyrite']
class_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 165, 0), (128, 0, 128), (255, 255, 0)]

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask).long()

        return image, mask

class Augmentation:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, mask):

        image = transforms.functional.resize(image, self.output_size)
        mask = transforms.functional.resize(mask, self.output_size, interpolation=InterpolationMode.NEAREST)


        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)


        if random.random() < 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)


        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)

        if random.random() < 0.5:
            kernel_size = random.choice([3, 5, 7])
            image = transforms.functional.gaussian_blur(image, kernel_size)


        if random.random() < 0.5:
            img_array = np.array(image)
            noise = np.random.poisson(img_array).astype(np.float32)
            noisy_img = np.clip(noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy_img)

        image = transforms.functional.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

def val_transform(image, mask, output_size):
    image = transforms.functional.resize(image, output_size)

    if mask is not None:
        mask = transforms.functional.resize(mask, output_size, interpolation=InterpolationMode.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()

    image = transforms.functional.to_tensor(image)

    if mask is not None:
        return image, mask
    else:
        return image

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) **self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice

def train(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device).squeeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def evaluate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    class_metrics = np.zeros((num_classes, 3))
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device).squeeze(1)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            for c in range(num_classes):
                true_positive = ((preds == c) & (masks == c)).sum().item()
                false_positive = ((preds == c) & (masks != c)).sum().item()
                false_negative = ((preds != c) & (masks == c)).sum().item()
                class_metrics[c] += [true_positive, false_positive, false_negative]
    class_ious = []
    for c in range(1, num_classes):
        tp, fp, fn = class_metrics[c]
        iou = tp / (tp + fp + fn + 1e-8)
        class_ious.append(iou)
    mean_iu = np.mean(class_ious) if class_ious else 0.0
    return running_loss / len(dataloader), mean_iu


class EnhancedBayesianOptimizer:
    def __init__(self, num_classes, train_loader, val_loader, device):
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device


        self.pbounds = {
            'ce_weight': (0.1, 0.9),
            'dice_weight': (0.1, 0.9),
            'sum_weights': (0.9, 1.1)
        }

        self.optimizer = BayesianOptimization(
            f=self._objective,
            pbounds=self.pbounds,
            random_state=42,
            allow_duplicate_points=True
        )

    def _normalize_weights(self, params):

        total = params['ce_weight'] + params['dice_weight']
        s = params['sum_weights']
        return {
            'ce_weight': params['ce_weight'] * s / total,
            'dice_weight': params['dice_weight'] * s / total
        }

    def _objective(self, ce_weight, dice_weight, sum_weights):
        try:

            params = {
                'ce_weight': ce_weight,
                'dice_weight': dice_weight,
                'sum_weights': sum_weights
            }
            norm_params = self._normalize_weights(params)
            ce_w = norm_params['ce_weight']
            dice_w = norm_params['dice_weight']


            max_epochs = int(3 + 5 * (ce_w + dice_w))


            model = MCCDSNet(num_classes=self.num_classes).to(self.device)
            criterion = CombinedLoss(ce_weight=ce_w, dice_weight=dice_w)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)


            best_loss = float('inf')
            patience = 2
            for epoch in range(max_epochs):
                model.train()
                total_loss = 0.0
                for images, masks in self.train_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks.squeeze(1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()


                avg_loss = total_loss / len(self.train_loader)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break


            model.eval()
            _, _, mean_iu, _, _ = evaluate(model, criterion, self.val_loader, self.device)
            return mean_iu

        except Exception as e:
            print(f"Optimization Error: {str(e)}")
            return 0

    def optimize(self, init_points=5, n_iter=25):
        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )


        best_raw = self.optimizer.max['params']
        best_params = self._normalize_weights(best_raw)
        print("\n=== Optimization Results ===")
        print(
            f"Original params: CE={best_raw['ce_weight']:.3f}, Dice={best_raw['dice_weight']:.3f}, Sum={best_raw['sum_weights']:.3f}")
        print(f"Normalized params: CE={best_params['ce_weight']:.4f}, Dice={best_params['dice_weight']:.4f}")
        return best_params



if __name__ == "__main__":

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(class_names)
    batch_size = 16
    learning_rate = 1e-2
    num_epochs = 100
    output_size = (512, 512)
    train_dataset = SegmentationDataset(
        image_dir='dataset/train/JPEGImages',
        mask_dir='dataset/train/SegmentationClass',
        transform=Augmentation(output_size)
    )
    val_dataset = SegmentationDataset(
        image_dir='dataset/val/JPEGImages',
        mask_dir='dataset/val/SegmentationClass',
        transform=lambda x, y: val_transform(x, y, output_size)
    )


    class_counts = torch.zeros(num_classes)
    for _, mask in train_dataset:
        class_counts += torch.bincount(mask.flatten(), minlength=num_classes)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    bayes_optimizer = EnhancedBayesianOptimizer(
        num_classes=num_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )


    best_params = bayes_optimizer.optimize(init_points=5, n_iter=25)
    best_ce_weight = best_params['ce_weight']
    best_dice_weight = best_params['dice_weight']

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)


    model = MCCDSNet(num_classes=num_classes).to(device)
    criterion = CombinedLoss(ce_weight=best_ce_weight, dice_weight=best_dice_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    results_df = pd.DataFrame(columns=[
        'Epoch',
        'Training Loss',
        'Validation Loss',
        'Validation Mean IoU'
    ])


    total_start_time = time.time()
    best_mean_iu = 0.0
    for epoch in range(num_epochs):
        train_loss = train(model, criterion, optimizer, train_loader, device)
        val_loss, val_miou = evaluate(model, criterion, val_loader, device)

        # 更新结果记录
        new_row = {
            'Epoch': epoch + 1,
            'Training Loss': train_loss,
            'Validation Loss': val_loss,
            'Validation Mean IoU': val_miou
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")


        if val_miou > best_mean_iu:
            best_mean_iu = val_miou
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'miou': val_miou
            }, "best_model.pth")
            print(f"Saved best model with mIoU {val_miou:.4f}")


        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch + 1}.pth")
            visualize_predictions(epoch + 1, model, device, output_size)
            results_df.to_excel("training_results.xlsx", index=False)
    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time / 3600:.2f} hours")
    results_df.to_excel("final_training_results.xlsx", index=False)
    torch.save(model.state_dict(), "final_model.pth")

plt.figure(figsize=(10, 6))
plt.plot(results_df['Epoch'], results_df['Training Loss'], label='Training Loss')
plt.plot(results_df['Epoch'], results_df['Validation Loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(results_df['Epoch'], results_df['Validation Mean IoU'], label='Validation mIoU', color='green')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.legend()
plt.savefig('miou_curve.png')
plt.close()
