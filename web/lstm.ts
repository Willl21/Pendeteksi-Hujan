// LSTM(64) -> Dense(2, sigmoid), sama persis dengan model_hujan.keras (urutan gate Keras: i, f, c, o).
// Ditulis manual karena hasil konversi TF.js untuk LSTM rusak / membengkak 13 MB.

export type BobotLSTM = {
  kernel: number[][];    // [fitur][4*unit]
  recurrent: number[][]; // [unit][4*unit]
  bias: number[];        // [4*unit]
  dense_w: number[][];   // [unit][2]
  dense_b: number[];     // [2]
};

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

/** x: [jam][fitur] yang sudah dinormalisasi. Hasil: [peluang hujan, peluang cerah]. */
export function jalankanLSTM(b: BobotLSTM, x: number[][]): number[] {
  const unit = b.recurrent.length;
  let h = new Array(unit).fill(0); // memori jangka pendek (output tiap langkah)
  let c = new Array(unit).fill(0); // memori jangka panjang (cell state)

  for (const xt of x) {
    // z = x_t·W + h·U + b, lalu dipecah jadi 4 gate
    const z = b.bias.slice();
    xt.forEach((v, i) => b.kernel[i].forEach((w, j) => (z[j] += v * w)));
    h.forEach((v, i) => b.recurrent[i].forEach((w, j) => (z[j] += v * w)));

    const hBaru = new Array(unit), cBaru = new Array(unit);
    for (let k = 0; k < unit; k++) {
      const masuk = sigmoid(z[k]);              // input gate: seberapa banyak info baru disimpan
      const lupa = sigmoid(z[unit + k]);        // forget gate: seberapa banyak memori lama dibuang
      const kandidat = Math.tanh(z[2 * unit + k]);
      const keluar = sigmoid(z[3 * unit + k]);  // output gate
      cBaru[k] = lupa * c[k] + masuk * kandidat;
      hBaru[k] = keluar * Math.tanh(cBaru[k]);
    }
    h = hBaru;
    c = cBaru;
  }
  // Dropout nggak aktif waktu prediksi, langsung ke Dense
  return b.dense_b.map((bias, j) => sigmoid(h.reduce((s, v, i) => s + v * b.dense_w[i][j], bias)));
}
