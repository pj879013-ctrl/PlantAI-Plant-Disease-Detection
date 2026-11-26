
import tensorflow as tf
import json
import numpy as np

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_PATH = "data/raw/archive/PlantVillage/PlantVillage"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 3
SEED = 42

MODEL_PATH = "models/plant_disease_model.keras"

print("\n======================================")
print("    PLANT DISEASE MODEL TRAINING")
print("======================================")

# ==========================================
# 2. LOAD DATASET
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nNumber of classes:", num_classes)

# ==========================================
# 3. CALCULATE CLASS WEIGHTS
# ==========================================

dataset_path = Path(DATASET_PATH)

class_counts = []

for class_name in class_names:
    folder = dataset_path / class_name

    count = sum(
        1 for file in folder.iterdir()
        if file.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    class_counts.append(count)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=np.repeat(
        np.arange(num_classes),
        class_counts
    )
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(class_weights_array)
}

# ==========================================
# 4. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
])

# ==========================================
# 5. BUILD MOBILENETV2 MODEL
# ==========================================

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False

inputs = layers.Input(shape=(160, 160, 3))

x = data_augmentation(inputs)

x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# ==========================================
# 6. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture created!")

model.summary(line_length=100)

# ==========================================
# 7. OPTIMIZE DATA PIPELINE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==========================================
# 8. TRAINING CALLBACKS
# ==========================================

Path("models").mkdir(exist_ok=True)

training_callbacks = [

    callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),

    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=1,
        min_lr=0.00001
    )
]

# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("\nStarting model training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=training_callbacks
)

# ==========================================
# 10. EVALUATE MODEL
# ==========================================

print("\n========== MODEL EVALUATION ==========")

best_model = tf.keras.models.load_model(MODEL_PATH)

loss, accuracy = best_model.evaluate(val_ds)

print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# 11. SAVE CLASS NAMES
# ==========================================

with open("models/class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)

print("\nModel saved:", MODEL_PATH)
print("Class names saved successfully!")

print("\n======================================")
print("       TRAINING COMPLETED")
print("======================================")

