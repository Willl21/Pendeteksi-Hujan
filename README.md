# Pendeteksi Cuaca

Web untuk memprediksi **hujan atau cerah dalam 12 jam ke depan**, menggabungkan dua model deep learning dengan prakiraan cuaca resmi:

1. **LSTM**: membaca data cuaca per jam selama 7 hari terakhir di lokasi pengguna, lalu menghitung peluang hujan dan peluang cerah.
2. **CNN (MobileNetV2)**: mengenali kondisi langit dari foto (opsional).
3. **Prakiraan Open-Meteo**: dipakai sebagai pembanding.

Kedua model berjalan **langsung di browser**, tanpa server. Foto pengguna tidak dikirim ke mana pun.

## Hasil model

| Model | Data | Hasil (data uji) | Pembanding |
|---|---|---|---|
| LSTM: hujan 12 jam | Jakarta + Bekasi, per jam, 2022–2026 | akurasi **73,0%**, AUC **0,804** | selalu mayoritas 54,4% · sama seperti 12 jam lalu 57,5% |
| LSTM: cerah 12 jam | sama | akurasi 89,1%, AUC **0,866** | selalu mayoritas 88,6% |
| MobileNetV2: foto langit | 3.123 foto, 7 kelas | akurasi validasi **95,0%** | – |

- Data uji LSTM adalah semua data **setelah 1 Juli 2025**, yang tidak pernah dipakai saat training.
- Untuk "cerah", akurasi menipu karena kelasnya jarang (±11%), sehingga model yang selalu menjawab "tidak cerah" pun sudah 88,6%. Karena itu yang dilihat adalah **AUC** (0,5 = asal tebak, 1,0 = sempurna), dan web menampilkan **persentase**, bukan jawaban ya/tidak.
- Kelas foto: berawan, kabut/asap, petir, hujan, pelangi, cerah, matahari terbit/terbenam. Kelas yang tidak relevan untuk Indonesia (salju, es, badai pasir, dll.) sengaja tidak dipakai. Dengan 11 kelas, akurasinya hanya 87%, dan foto langit mendung sempat ditebak "badai pasir".

## Cara kerja

```
Lokasi (GPS / Jakarta / Bekasi)
  └─ Open-Meteo: 168 jam terakhir + 12 jam prakiraan
       ├─ LSTM (web/lstm.ts) ───────────► peluang hujan & cerah   bobot 40%
       └─ prakiraan Open-Meteo ─────────► peluang hujan & awan    bobot 40%
Foto langit (opsional)
  └─ MobileNetV2 (TF.js) ───────────────► kelas langit sekarang   bobot 20%
                                   │
                         rata-rata berbobot ─► kesimpulan + warna tema
```

- **Fitur LSTM** (10 per jam): suhu, kelembapan, tekanan, tutupan awan, kecepatan angin, curah hujan (`log1p`), jam dan hari-dalam-tahun (sin/cos).
- **Rumus LSTM ditulis manual** di `web/lstm.ts`, karena hasil konversi TF.js untuk LSTM rusak, dan kalau diakali ukurannya membengkak jadi 13 MB. `npm test` membuktikan hasilnya identik dengan Keras.
- **Bobot penggabungan ditentukan manual**, bukan hasil training, karena tidak ada data foto yang punya data sensor di waktu dan tempat yang sama.

## Struktur

| Path | Isi |
|---|---|
| `cuaca_colab.ipynb` | training model foto (Google Colab, GPU) |
| `hujan_lstm.ipynb` | training LSTM; data diambil langsung dari Open-Meteo |
| `model_cuaca.keras`, `labels.json` | model foto hasil training |
| `model_hujan.keras`, `hujan_config.json` | LSTM + angka normalisasi fitur |
| `ekspor_web.py` | konversi model ke `public/model/` untuk web |
| `cuaca.py` | prediksi foto dari terminal: `python cuaca.py predict foto.jpg` |
| `web/` | aplikasi web (Vite + React + TypeScript) |

## Menjalankan

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # cek LSTM & fitur web sama dengan Keras/notebook
npm run deploy     # build + deploy ke Cloudflare Workers
```

**Training ulang:**
1. Jalankan notebook di Google Colab (Runtime → T4 GPU). Notebook foto butuh `data.zip` di Google Drive.
2. Taruh file `.keras` dan `.json` hasil download di folder ini.
3. Jalankan `python ekspor_web.py`. Script ini butuh konverter TF.js tanpa dependensinya:
   `pip install --no-deps tensorflowjs==4.22.0 tf-keras==<versi TensorFlow> tensorflow-hub importlib_resources`

## Keterbatasan

- LSTM hanya dilatih dengan data **Jakarta dan Bekasi**. Di kota lain hasilnya kurang bisa dipercaya.
- Label hujan berasal dari **data model cuaca Open-Meteo**, bukan alat ukur hujan di lapangan. Jadi model belajar meniru pola hujan versi model cuaca.
- Foto dataset diambil dari internet. Pada foto HP sungguhan, akurasinya kemungkinan lebih rendah. Selain itu, kelas berawan/cerah/matahari terbit hanya berasal dari satu dataset, sehingga model mungkin ikut mempelajari "gaya foto" dataset itu.
- Prakiraan resmi Open-Meteo (model cuaca numerik) tetap lebih akurat daripada LSTM ini. LSTM di proyek ini bertujuan pembelajaran.

## Sumber data

- **WEAPD**: Xiao, H. (2021). *Weather phenomenon database (WEAPD)*. Harvard Dataverse. [doi:10.7910/DVN/M8JQCR](https://doi.org/10.7910/DVN/M8JQCR). Lisensi CC0.
- **Multi-class Weather Dataset for Image Classification**: Ajayi, G. (2018). Mendeley Data, V1. [doi:10.17632/4drtyfjtfy.1](https://doi.org/10.17632/4drtyfjtfy.1). Lisensi [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Dataset ini digabung dengan WEAPD, dan gambarnya diperkecil ke maksimal 512 px serta dikonversi ke JPEG.
- **Open-Meteo**: data cuaca per jam dan prakiraan ([open-meteo.com](https://open-meteo.com), CC BY 4.0).
- Ikon: [Lucide](https://lucide.dev).
