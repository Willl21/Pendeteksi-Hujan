"""Pendeteksi cuaca dari gambar (transfer learning MobileNetV2).

Dataset: data/<kelas>/*.jpg, contoh data/cerah, data/hujan, data/berawan, ...
Train:   python cuaca.py train [folder_data] [epochs]
Prediksi: python cuaca.py predict foto.jpg
"""
import json
import sys

import keras
import numpy as np

SIZE = (224, 224)
MODEL, LABELS = "model_cuaca.keras", "labels.json"


def train(data_dir="data", epochs=10):
    kw = dict(validation_split=0.2, seed=42, image_size=SIZE, batch_size=32)
    train_ds = keras.utils.image_dataset_from_directory(data_dir, subset="training", **kw)
    val_ds = keras.utils.image_dataset_from_directory(data_dir, subset="validation", **kw)
    classes = train_ds.class_names

    base = keras.applications.MobileNetV2(input_shape=SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False  # ponytail: backbone frozen; unfreeze top layers + low LR if accuracy plateaus
    model = keras.Sequential([
        keras.Input(SIZE + (3,)),
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.Rescaling(1 / 127.5, offset=-1),  # MobileNetV2 expects [-1, 1]
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(len(classes), activation="softmax"),
    ])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    model.save(MODEL)
    json.dump(classes, open(LABELS, "w"))
    print(f"Tersimpan: {MODEL}, kelas: {classes}")


def predict(path):
    model = keras.models.load_model(MODEL)
    classes = json.load(open(LABELS))
    img = keras.utils.img_to_array(keras.utils.load_img(path, target_size=SIZE))
    probs = model.predict(img[None], verbose=0)[0]
    for i in np.argsort(probs)[::-1]:
        print(f"{classes[i]:>10}: {probs[i]:.1%}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["help"]
    if cmd == "train":
        train(*args[:1], *(int(a) for a in args[1:2]))
    elif cmd == "predict" and args:
        predict(args[0])
    else:
        print(__doc__)
