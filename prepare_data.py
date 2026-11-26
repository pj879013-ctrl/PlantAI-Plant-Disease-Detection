
import tensorflow as tf
import json
from pathlib import Path

# =====================================
# CONFIGURATION
# =====================================

DATASET_PATH = (
    "data/raw/archive/PlantVillage/PlantVillage"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.20

print("\n====================================")
print("     PLANT DISEASE DATA PREPARATION")
print("====================================\n")

# =====================================
# LOAD TRAINING DATA
# =====================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

# =====================================
# LOAD VALIDATION DATA
# =====================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

# =====================================
# SAVE CLASS NAMES
# =====================================

class_names = train_dataset.class_names

Path("models").mkdir(exist_ok=True)

with open("models/class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)

# =====================================
# NORMALIZE IMAGES
# =====================================

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda images, labels: (
        normalization_layer(images), labels
    )
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (
        normalization_layer(images), labels
    )
)

# =====================================
# OPTIMIZE DATA PIPELINE
# =====================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# =====================================
# DISPLAY RESULTS
# =====================================

print("\n========== DATASET SUMMARY ==========")

print("Number of classes:", len(class_names))

print("\nDisease categories:")

for index, name in enumerate(class_names):
    print(f"{index}: {name}")

print("\nTraining batches:", len(train_dataset))
print("Validation batches:", len(validation_dataset))

for images, labels in train_dataset.take(1):
    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Pixel value range:",
          float(tf.reduce_min(images)),
          "to",
          float(tf.reduce_max(images)))

print("\nClass names saved successfully!")
print("====================================")
print("DATA PREPARATION COMPLETED")
print("====================================")
