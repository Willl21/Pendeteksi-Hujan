// Dimuat terpisah (import dinamis) biar TF.js ±1 MB cuma diunduh kalau user pilih foto
import * as tf from "@tensorflow/tfjs";

export const muatModelFoto = () =>
  Promise.all([tf.loadGraphModel("/model/cuaca/model.json"), fetch("/model/labels.json").then((r) => r.json() as Promise<string[]>)]);

/** Foto -> peluang tiap kelas. Model sudah berisi Rescaling, jadi input-nya piksel 0–255. */
export async function prediksiFoto(model: tf.GraphModel, labels: string[], gambar: HTMLImageElement) {
  const kanvas = document.createElement("canvas");
  kanvas.width = kanvas.height = 224;
  kanvas.getContext("2d")!.drawImage(gambar, 0, 0, 224, 224); // kecilin dulu biar foto 12 MP nggak makan RAM HP
  const p = tf.tidy(() => model.predict(tf.browser.fromPixels(kanvas).toFloat().expandDims(0)) as tf.Tensor);
  const nilai = Array.from(await p.data());
  p.dispose();
  return Object.fromEntries(labels.map((l, i) => [l, nilai[i]]));
}
