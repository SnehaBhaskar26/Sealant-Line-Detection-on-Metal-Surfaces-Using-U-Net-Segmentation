import os
import shutil
import random
from pathlib import Path

def split_dataset(
    images_dir,
    masks_dir,
    output_dir='dataset_split',
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,
    image_ext=".jpg",
    mask_ext=".png"
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    random.seed(seed)
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    # Collect image files with given extension
    image_files = sorted([f for f in images_dir.glob(f"*{image_ext}") if f.is_file()])
    print(f"Found {len(image_files)} image files.")
    
    if len(image_files) == 0:
        print("❌ No image files found. Check your 'images_dir' and file extensions.")
        return

    random.shuffle(image_files)

    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    for split, files in splits.items():
        img_out = output_dir / split / 'images'
        mask_out = output_dir / split / 'masks'
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            # Construct corresponding mask path (change extension if needed)
            mask_path = masks_dir / (img_path.stem + mask_ext)
            if not mask_path.exists():
                print(f"⚠️  Warning: Mask not found for {img_path.name}, skipping...")
                continue
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(mask_path, mask_out / mask_path.name)

    print("\n✅ Dataset split complete!\n")
    for split in splits:
        count = len(list((output_dir / split / 'images').glob("*")))
        print(f"{split.upper()}: {count} images")

# Example usage
if __name__ == "__main__":
    split_dataset(
        images_dir="/home/sneha/Selent/final_sealant/401",
        masks_dir="/home/sneha/Selent/final_sealant/mask_401",
        output_dir="split_dataset",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        image_ext=".jpg",   # Adjust as needed
        mask_ext=".png"     # Adjust as needed
    )
