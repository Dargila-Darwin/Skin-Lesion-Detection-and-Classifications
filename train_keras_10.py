import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# =========================================================
# CLASS NAMES (11 classes)
# =========================================================

CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Chickenpox",
    "Cowpox",
    "Dermatofibroma",
    "Healthy",
    "HFMD",
    "Measles",
    "Melanocytic nevi",
    "unknown"
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =========================================================
# COMPUTE CLASS WEIGHTS
# =========================================================

def compute_class_weights(train_dir):

    counts = []

    for cname in CLASS_NAMES:

        cdir = train_dir / cname

        count = len([p for p in cdir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

        counts.append(count)

    total = sum(counts)

    weights = {}

    for i, c in enumerate(counts):

        weights[i] = total / (len(CLASS_NAMES) * c)

    print("\nClass Distribution")

    for cname, count in zip(CLASS_NAMES, counts):
        print(cname, ":", count)

    print("\nClass Weights:", weights)

    return weights


# =========================================================
# DATASET LOADER
# =========================================================

def make_datasets(train_dir, val_dir, img_size, batch_size):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


# =========================================================
# MODEL
# =========================================================

def build_model(img_size, num_classes):

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.2),
    ])

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    x = data_augmentation(inputs)

    x = tf.keras.applications.efficientnet.preprocess_input(x)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    return model, base_model


# =========================================================
# EVALUATION
# =========================================================

def evaluate(model, dataset):

    y_true = []
    y_pred = []

    for x, y in dataset:

        preds = model.predict(x, verbose=0)

        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("\nClassification Report\n")

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix\n", cm)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(model, image_path, img_size=224):

    img = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))

    img = tf.keras.utils.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.efficientnet.preprocess_input(img)

    preds = model.predict(img)[0]

    class_index = np.argmax(preds)

    label = CLASS_NAMES[class_index]

    confidence = preds[class_index]

    print("\nPrediction:", label)

    print("Confidence:", confidence)

    return label, confidence


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", type=Path, default=Path("train_10class"))
    parser.add_argument("--val-dir", type=Path, default=Path("val_10class"))
    parser.add_argument("--test-dir", type=Path, default=Path("test_10class"))

    parser.add_argument("--img-size", type=int, default=224)

    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--epochs-head", type=int, default=6)

    parser.add_argument("--epochs-ft", type=int, default=14)

    parser.add_argument("--fine-tune-at", type=int, default=100)

    parser.add_argument("--out", type=Path, default=Path("models/efficientnet_skin.keras"))

    args = parser.parse_args()

    train_ds, val_ds = make_datasets(
        args.train_dir,
        args.val_dir,
        args.img_size,
        args.batch_size
    )

    class_weights = compute_class_weights(args.train_dir)

    model, base_model = build_model(args.img_size, len(CLASS_NAMES))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(str(args.out), save_best_only=True),

        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),

        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.3)
    ]

    print("\nTraining Head")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # ========================
    # Fine Tuning
    # ========================

    base_model.trainable = True

    for layer in base_model.layers[:args.fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nFine Tuning")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head + args.epochs_ft,
        initial_epoch=args.epochs_head,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # ========================
    # TEST EVALUATION
    # ========================

    if args.test_dir.exists():

        test_ds = tf.keras.utils.image_dataset_from_directory(
            args.test_dir,
            label_mode="categorical",
            class_names=CLASS_NAMES,
            image_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            shuffle=False
        )

        evaluate(model, test_ds)

    model.save(args.out)

    print("\nModel saved at:", args.out)


if __name__ == "__main__":
    main()