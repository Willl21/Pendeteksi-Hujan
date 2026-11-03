"""Ekspor model hasil Colab ke folder web (public/model/). Jalankan ulang tiap model di-training ulang.

    python ekspor_web.py

Butuh konverter TF.js tanpa dependensinya (tensorflow-decision-forests nggak ada versi Windows):
    pip install --no-deps tensorflowjs==4.22.0 tf-keras==<versi TensorFlow kamu> tensorflow-hub importlib_resources
"""
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import keras
import numpy as np

# tensorflowjs import-paksa paket ini, padahal cuma dipakai untuk model JAX / decision forest
for nama in ["jax", "jax.experimental", "jax.experimental.jax2tf", "tensorflow_decision_forests"]:
    sys.modules.setdefault(nama, types.ModuleType(nama))
sys.modules["jax"].experimental = sys.modules["jax.experimental"]
sys.modules["jax.experimental"].jax2tf = sys.modules["jax.experimental.jax2tf"]
from tensorflowjs.converters import tf_saved_model_conversion_v2 as tfjs  # noqa: E402

OUT = Path("public/model")
OUT.mkdir(parents=True, exist_ok=True)

# 1. Model gambar -> TF.js graph model. Layer augmentasi (Random*) cuma aktif waktu training
#    dan bikin export gagal, jadi dibuang; bobot lainnya tetap sama.
gambar = keras.models.load_model("model_cuaca.keras")
lapis = [l for l in gambar.layers if not type(l).__name__.startswith("Random")]
gambar = keras.Sequential([keras.Input(gambar.input_shape[1:])] + lapis)
with tempfile.TemporaryDirectory() as tmp:
    gambar.export(tmp)
    shutil.rmtree(OUT / "cuaca", ignore_errors=True)
    tfjs.convert_tf_saved_model(tmp, str(OUT / "cuaca"))
# Download manager (mis. IDM) mencegat file .bin dan ngasih browser respon kosong -> ganti ekstensi
model_json = json.load(open(OUT / "cuaca" / "model.json"))
for grup in model_json["weightsManifest"]:
    for i, nama in enumerate(grup["paths"]):
        grup["paths"][i] = nama.replace(".bin", ".bobot")
        (OUT / "cuaca" / nama).rename(OUT / "cuaca" / grup["paths"][i])
json.dump(model_json, open(OUT / "cuaca" / "model.json", "w"))
shutil.copy("labels.json", OUT / "labels.json")

# 2. LSTM -> bobot mentah (JSON). Konversi TF.js bikin loop LSTM rusak / membengkak 13 MB,
#    jadi rumus LSTM-nya ditulis langsung di web/lstm.ts (cuma ±150 KB bobot).
hujan = keras.models.load_model("model_hujan.keras")
lstm, dense = hujan.layers[0], hujan.layers[-1]
assert type(lstm).__name__ == "LSTM" and lstm.activation.__name__ == "tanh", "web/lstm.ts cuma ngerti LSTM standar"
kernel, recurrent, bias = lstm.get_weights()
dense_w, dense_b = dense.get_weights()
bulat = lambda a: np.round(a, 7).tolist()
json.dump(dict(kernel=bulat(kernel), recurrent=bulat(recurrent), bias=bulat(bias),
               dense_w=bulat(dense_w), dense_b=bulat(dense_b)), open(OUT / "lstm_bobot.json", "w"))
shutil.copy("hujan_config.json", OUT / "hujan_config.json")

# 3. Contoh input + jawaban Keras, buat ngetes web/lstm.ts (npm test)
x = np.random.default_rng(0).normal(size=(1, 168, kernel.shape[0])).astype("float32")
json.dump(dict(x=bulat(x[0]), y=hujan.predict(x, verbose=0)[0].tolist()), open("web/lstm.fixture.json", "w"))
print("Selesai ->", OUT)
