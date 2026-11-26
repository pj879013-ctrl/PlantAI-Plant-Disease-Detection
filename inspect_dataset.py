
from pathlib import Path

DATASET_PATH = Path(
    "data/raw/archive/PlantVillage/PlantVillage"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

total_images = 0
total_classes = 0

print("\n===== FINAL DATASET VERIFICATION =====\n")

for folder in sorted(DATASET_PATH.iterdir()):

    if folder.is_dir():

        images = [
            f for f in folder.iterdir()
            if f.is_file()
            and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if images:
            total_classes += 1
            total_images += len(images)

            print(f"{folder.name}: {len(images)} images")

print("\n================================")
print("Total classes:", total_classes)
print("Total images:", total_images)
print("================================")
