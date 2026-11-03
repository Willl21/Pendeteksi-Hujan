import { jalankanLSTM, type BobotLSTM } from "./lstm.ts";

export type Config = { fitur: string[]; mean: number[]; std: number[]; jendela: number; ke_depan: number };
type Jam = { time: string[] } & Record<string, (number | null)[]>;

const VARS = ["temperature_2m", "relative_humidity_2m", "pressure_msl", "cloud_cover", "wind_speed_10m", "precipitation"];

// ---------- Data Open-Meteo ----------

export async function ambilCuaca(lat: number, lon: number) {
  const q = new URLSearchParams({
    latitude: String(lat), longitude: String(lon), timezone: "auto", past_days: "8", forecast_days: "2",
    hourly: [...VARS, "precipitation_probability"].join(","),
  });
  const res = await fetch(`https://api.open-meteo.com/v1/forecast?${q}`);
  if (!res.ok) throw new Error(`Open-Meteo error ${res.status}`);
  const d = await res.json();
  // Jam lokal lokasi itu, format sama dengan d.hourly.time ("2026-11-03T14:37")
  const sekarang = new Date(Date.now() + d.utc_offset_seconds * 1000).toISOString().slice(0, 16);
  return { jam: d.hourly as Jam, sekarang };
}

/** Sama dengan fitur() di hujan_lstm.ipynb: 6 variabel (hujan di-log1p) + jam & hari dalam sin/cos. */
export function fitur(jam: Jam, i: number): number[] {
  const [tgl, waktu] = jam.time[i].split("T");
  const [y, m, d] = tgl.split("-").map(Number);
  const j = Number(waktu.slice(0, 2));
  const hari = (Date.UTC(y, m - 1, d) - Date.UTC(y, 0, 0)) / 864e5; // 1 = 1 Januari
  const nilai = VARS.map((v) => {
    // ponytail: data kosong diisi nilai jam sebelumnya (notebook pakai interpolasi linear); jarang terjadi
    let k = i;
    while (k > 0 && jam[v][k] == null) k--;
    const x = jam[v][k] ?? 0;
    return v === "precipitation" ? Math.log1p(x) : x;
  });
  return [...nilai, Math.sin((2 * Math.PI * j) / 24), Math.cos((2 * Math.PI * j) / 24),
    Math.sin((2 * Math.PI * hari) / 365.25), Math.cos((2 * Math.PI * hari) / 365.25)];
}

/** Peluang hujan & cerah 12 jam ke depan dari 168 jam terakhir, plus prakiraan Open-Meteo sebagai pembanding. */
export function prediksiHujan(jam: Jam, sekarang: string, cfg: Config, bobot: BobotLSTM) {
  const akhir = jam.time.findLastIndex((t) => t <= sekarang); // jam ini ikut, sama seperti notebook
  if (akhir + 1 < cfg.jendela) throw new Error("Data 7 hari terakhir kurang");
  const x = [];
  for (let i = akhir + 1 - cfg.jendela; i <= akhir; i++)
    x.push(fitur(jam, i).map((v, k) => (v - cfg.mean[k]) / cfg.std[k]));
  const [hujan, cerah] = jalankanLSTM(bobot, x);

  const depan = (v: string) => jam[v].slice(akhir + 1, akhir + 1 + cfg.ke_depan).map((n) => n ?? 0);
  return {
    hujan, cerah,
    openMeteo: {
      hujanMm: depan("precipitation").reduce((a, b) => a + b, 0),
      peluangHujan: Math.max(...depan("precipitation_probability")) / 100,
      awan: depan("cloud_cover").reduce((a, b) => a + b, 0) / cfg.ke_depan,
    },
  };
}

// ---------- Gabungan ----------

// Kelas foto dikelompokkan ke hujan / cerah; sisanya (cloudy, fogsmog, rainbow) dianggap netral.
const FOTO_HUJAN = ["rain", "lightning"], FOTO_CERAH = ["shine", "sunrise"];
// ponytail: bobot ditentukan manual, bukan hasil training (nggak ada data foto + sensor berpasangan)
const BOBOT = { lstm: 0.4, openMeteo: 0.4, foto: 0.2 };

export function gabungkan(h: ReturnType<typeof prediksiHujan>, foto?: Record<string, number>) {
  const om = h.openMeteo;
  const omCerah = om.hujanMm < 0.1 ? Math.max(0, 1 - om.awan / 60) : 0; // awan 0% -> 1, >= 60% -> 0
  const sumber = [
    { bobot: BOBOT.lstm, hujan: h.hujan, cerah: h.cerah },
    { bobot: BOBOT.openMeteo, hujan: om.peluangHujan, cerah: omCerah },
  ];
  if (foto) {
    const jumlah = (k: string[]) => k.reduce((s, l) => s + (foto[l] ?? 0), 0);
    sumber.push({ bobot: BOBOT.foto, hujan: jumlah(FOTO_HUJAN), cerah: jumlah(FOTO_CERAH) });
  }
  const total = sumber.reduce((s, x) => s + x.bobot, 0);
  const rata = (k: "hujan" | "cerah") => sumber.reduce((s, x) => s + x.bobot * x[k], 0) / total;
  const hujan = rata("hujan"), cerah = rata("cerah");
  const beda = Math.max(...sumber.map((s) => s.hujan)) - Math.min(...sumber.map((s) => s.hujan));

  const kesimpulan =
    hujan >= 0.6 ? "Kemungkinan besar hujan" :
    cerah >= 0.5 ? "Kemungkinan cerah" :
    hujan < 0.35 ? "Berawan, kemungkinan tidak hujan" :
    "Belum pasti, bisa hujan bisa tidak";
  return { hujan, cerah, kesimpulan, sepakat: beda < 0.3 };
}
