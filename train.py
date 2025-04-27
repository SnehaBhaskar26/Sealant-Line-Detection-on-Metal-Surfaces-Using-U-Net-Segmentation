import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.metrics import jaccard_score
import numpy as np

# -------------------------------
#           CONFIG
# -------------------------------
IMAGE_DIR = "/home/eternal/sneha/final_sealant/split_dataset/train/images"
MASK_DIR = "/home/eternal/sneha/final_sealant/split_dataset/train/masks"
# VAL_IMAGE_DIR = "/home/sneha/Selent/data/output_split_dataset/val/images"
# VAL_MASK_DIR = "/home/sneha/Selent/data/output_split_dataset/val/masks"
IMAGE_SIZE = (1024, 1024)
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_PATH = "./checkpoints/unet_epoch25.pth"
RESULT_DIR = "./evaluation_results"
TEST_IMAGE_PATH = "/home/eternal/sneha/final_sealant/split_dataset/test/images/9742_-_401_cam1_20.jpg"
THRESHOLD = 0.5

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------------
#       U-Net Architecture
# -------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        u1 = self.up1(d4)
        u1 = self.conv1(torch.cat([u1, d3], dim=1))
        u2 = self.up2(u1)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))
        u3 = self.up3(u2)
        u3 = self.conv3(torch.cat([u3, d1], dim=1))

        return torch.sigmoid(self.output(u3))

# -------------------------------
#     Custom Dataset Class
# -------------------------------
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.pairs = []

        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = base_name + ".png"
            mask_path = os.path.join(mask_dir, mask_file)
            if os.path.exists(mask_path):
                self.pairs.append((os.path.join(image_dir, img_file), mask_path))
            else:
                print(f"⚠️ Warning: No mask found for {img_file}, skipping.")

        if len(self.pairs) == 0:
            raise RuntimeError("❌ No valid image-mask pairs found!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# -------------------------------
#       Dice Score Function
# -------------------------------
def dice_score(pred, target):
    smooth = 1e-6
    pred = (pred > THRESHOLD).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

# -------------------------------
#          Training
# -------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=False)

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch}: Loss = {avg_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"unet_epoch{epoch}.pth"))

# -------------------------------
#         Evaluation Code
# -------------------------------
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    dataset = SegmentationDataset(VAL_IMAGE_DIR, VAL_MASK_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0
    sample_counter = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            # outputs = model(images)
            # preds = (outputs > THRESHOLD).float()
            with torch.no_grad():
                preds = model(images)
            preds = (preds > 0.5).float() 

            for p, m in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                p = p.squeeze().astype(np.uint8)
                m = m.squeeze().astype(np.uint8)
                total_iou += jaccard_score(m.flatten(), p.flatten(), average='binary')


            preds_np = preds.cpu().numpy().reshape(images.size(0), -1)
            masks_np = masks.cpu().numpy().reshape(images.size(0), -1)
            for p, m in zip(preds_np, masks_np):
                total_iou += jaccard_score(m, p, average='binary')

            for i in range(images.size(0)):
                save_image(images[i], f"{RESULT_DIR}/sample_{sample_counter}_input.png")
                save_image(masks[i], f"{RESULT_DIR}/sample_{sample_counter}_mask.png")
                save_image(preds[i], f"{RESULT_DIR}/sample_{sample_counter}_pred.png")
                sample_counter += 1
                if sample_counter >= 10:
                    break

            total_samples += images.size(0)
            if sample_counter >= 10:
                break

    avg_dice = total_dice / total_samples
    avg_iou = total_iou / total_samples
    print(f"\n📊 Evaluation Metrics:")
    print(f"✅ Dice Score: {avg_dice:.4f}")
    print(f"✅ IoU Score : {avg_iou:.4f}")
    print(f"🖼️ Saved visualizations in `{RESULT_DIR}/`")

# -------------------------------
#     Test Single Image
# -------------------------------
def test_single_image(image_path=TEST_IMAGE_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load original image and its original size
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    # Preprocess for model input
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = (output > THRESHOLD).float().squeeze().cpu().numpy()

    # Resize prediction mask back to original image size
    mask_resized = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize(original_size, resample=Image.NEAREST)
    binary_mask = np.array(mask_resized) > (THRESHOLD * 255)

    # Extract masked region
    original_np = np.array(image)
    masked_np = original_np.copy()
    masked_np[~binary_mask] = 0  # Zero out background

    # Save files
    image.save(os.path.join(RESULT_DIR, "test_input_original.png"))
    mask_resized.save(os.path.join(RESULT_DIR, "test_prediction_mask.png"))
    Image.fromarray(masked_np).save(os.path.join(RESULT_DIR, "test_masked_region.png"))

    print("🧪 Inference done. Saved:")
    print("   - test_input_original.png")
    print("   - test_prediction_mask.png")
    print("   - test_masked_region.png")
    
    # print("🧪 Inference done. Results saved to evaluation_results/")

# -------------------------------
#     Test Image Folder
# -------------------------------
def test_folder_images(test_folder="/home/eternal/sneha/final_sealant/split_dataset/test/images"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    model = UNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for idx, filename in enumerate(tqdm(image_files, desc="Testing")):
        image_path = os.path.join(test_folder, filename)
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = (output > THRESHOLD).float().squeeze().cpu().numpy()
        # Resize prediction mask back to original image size
        mask_resized = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize(original_size, resample=Image.NEAREST)
        binary_mask = np.array(mask_resized) > (THRESHOLD * 255)
        # Extract masked region
        original_np = np.array(image)
        masked_np = original_np.copy()
        masked_np[~binary_mask] = 0  # Zero out background
        base_name = os.path.splitext(filename)[0]
        image.save(os.path.join(RESULT_DIR, f"{base_name}_input.png"))
        mask_resized.save(os.path.join(RESULT_DIR, f"{base_name}_pred_mask.png"))
        Image.fromarray(masked_np).save(os.path.join(RESULT_DIR, f"{base_name}_masked.png"))
    print(f":test_tube: Inference completed for folder `{test_folder}`. Results saved in `{RESULT_DIR}/`")




# -------------------------------
#               MAIN
# -------------------------------

if __name__ == "__main__":
    mode = input("Enter mode (`train`, `evaluate`, `test`, `test_folder`): ").strip().lower()
    if mode == "train":
        train()
    elif mode == "evaluate":
        evaluate()
    elif mode == "test":
        test_single_image()
    elif mode == "test_folder":
        test_folder_images()
    else:
        print(":x: Invalid mode. Please choose `train`, `evaluate`, `test`, or `test_folder`.")
