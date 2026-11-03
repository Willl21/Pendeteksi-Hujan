import { StrictMode, useEffect, useState, type ChangeEvent } from "react";
import { createRoot } from "react-dom/client";
import {
  BrainCircuit, Camera, CircleCheck, Clock, Cloud, CloudFog, CloudLightning, CloudMoon, CloudRain, CloudSun,
  Droplets, LoaderCircle, LocateFixed, MapPin, Moon, Rainbow, Satellite, Sun, Sunrise, TriangleAlert, type LucideIcon,
} from "lucide-react";
import { ambilCuaca, gabungkan, prediksiHujan, type Config } from "./cuaca.ts";
import type { BobotLSTM } from "./lstm.ts";
import "./style.css";

const KELAS: Record<string, { nama: string; ikon: LucideIcon }> = {
  cloudy: { nama: "Berawan", ikon: Cloud }, fogsmog: { nama: "Kabut/asap", ikon: CloudFog },
  lightning: { nama: "Petir", ikon: CloudLightning }, rain: { nama: "Hujan", ikon: CloudRain },
  rainbow: { nama: "Pelangi", ikon: Rainbow }, shine: { nama: "Cerah", ikon: Sun },
  sunrise: { nama: "Matahari terbit/terbenam", ikon: Sunrise },
};
const IKON_TEMA: Record<string, LucideIcon> = { hujan: CloudRain, cerah: Sun, berawan: Cloud, ragu: CloudSun, awal: CloudSun };
// Warna langit tiap tema; dipakai juga buat warna status bar HP
const LANGIT: Record<string, string> = { hujan: "#c5d4ea", cerah: "#fde2a0", berawan: "#e3e8ef", ragu: "#d9edf4", awal: "#dcf0fb" };
const KOTA = { Jakarta: [-6.2, 106.85], Bekasi: [-6.24, 106.99] } as const;

const persen = (v: number) => `${Math.round(v * 100)}%`;
const json = (p: string) => fetch(p).then((r) => r.json());

// Model hujan kecil (±400 KB) langsung dimuat; model foto (±9 MB) baru dimuat waktu ada foto
const modelHujan = Promise.all([json("/model/hujan_config.json"), json("/model/lstm_bobot.json")]) as Promise<[Config, BobotLSTM]>;
let modelFoto: Promise<[typeof import("./foto.ts"), Awaited<ReturnType<typeof import("./foto.ts").muatModelFoto>>]> | undefined;

type Prediksi = ReturnType<typeof prediksiHujan>;

function ikonJam(p: Prediksi["perJam"][number]): LucideIcon {
  const malam = p.jam >= 18 || p.jam < 6;
  if (p.peluang >= 0.5) return CloudRain;
  if (p.awan >= 70) return Cloud;
  if (p.awan >= 30) return malam ? CloudMoon : CloudSun;
  return malam ? Moon : Sun;
}

function Bar({ label, nilai }: { label: string; nilai: number }) {
  return (
    <div className="bar">
      <span>{label}</span>
      <div className="isi" role="meter" aria-valuenow={Math.round(nilai * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
        <div style={{ transform: `scaleX(${nilai})` }} />
      </div>
      <span>{persen(nilai)}</span>
    </div>
  );
}

function App() {
  const [lokasi, setLokasi] = useState<{ nama: string; lat: number; lon: number } | null>(null);
  const [hujan, setHujan] = useState<Prediksi | null>(null);
  const [foto, setFoto] = useState<{ url: string; hasil: Record<string, number> } | null>(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!lokasi) return;
    let batal = false;
    setHujan(null); setError(""); setStatus("Mengambil data cuaca 7 hari terakhir…");
    (async () => {
      const [{ jam, sekarang }, [cfg, bobot]] = await Promise.all([ambilCuaca(lokasi.lat, lokasi.lon), modelHujan]);
      if (!batal) setHujan(prediksiHujan(jam, sekarang, cfg, bobot));
    })().catch((e) => !batal && setError(String(e.message ?? e))).finally(() => !batal && setStatus(""));
    return () => { batal = true; };
  }, [lokasi]);

  const gabungan = hujan && gabungkan(hujan, foto?.hasil);
  const tema = gabungan?.tema ?? "awal";

  useEffect(() => {
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", LANGIT[tema]);
  }, [tema]);

  function pakaiGPS() {
    if (!navigator.geolocation) return setError("Browser ini nggak bisa baca lokasi. Pilih Jakarta atau Bekasi.");
    setError(""); setStatus("Minta izin lokasi…");
    navigator.geolocation.getCurrentPosition(
      (p) => setLokasi({ nama: "Lokasi kamu", lat: p.coords.latitude, lon: p.coords.longitude }),
      () => { setStatus(""); setError("Izin lokasi ditolak. Pilih Jakarta atau Bekasi."); },
      { timeout: 15000, maximumAge: 600000 },
    );
  }

  async function pilihFoto(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ""; // biar foto yang sama bisa dipilih lagi
    if (!file) return;
    const url = URL.createObjectURL(file);
    setError(""); setStatus("Menganalisis foto…");
    try {
      modelFoto ??= import("./foto.ts").then(async (m) => [m, await m.muatModelFoto()]);
      const [{ prediksiFoto }, [model, labels]] = await modelFoto;
      const img = new Image();
      img.src = url;
      await img.decode();
      if (foto) URL.revokeObjectURL(foto.url);
      setFoto({ url, hasil: await prediksiFoto(model, labels, img) });
    } catch (err) {
      modelFoto = undefined;
      URL.revokeObjectURL(url);
      setError("Gagal menganalisis foto: " + String((err as Error).message ?? err));
    } finally {
      setStatus("");
    }
  }

  const IkonUtama = IKON_TEMA[tema];
  const fotoUrut = foto && Object.entries(foto.hasil).sort((a, b) => b[1] - a[1]);
  const fotoAtas = fotoUrut && (KELAS[fotoUrut[0][0]] ?? { nama: fotoUrut[0][0], ikon: Camera });

  return (
    <div className="app" data-tema={tema} style={{ ["--langit" as string]: LANGIT[tema] }}>
      <nav className="pilih-lokasi" aria-label="Pilih lokasi">
        <button className="pil" onClick={pakaiGPS}><LocateFixed size={18} aria-hidden /> Lokasi saya</button>
        {Object.entries(KOTA).map(([nama, [lat, lon]]) => (
          <button key={nama} className="pil" aria-pressed={lokasi?.nama === nama} onClick={() => setLokasi({ nama, lat, lon })}>{nama}</button>
        ))}
      </nav>

      <header className="hero">
        <p className="nama-lokasi"><MapPin size={16} aria-hidden /> {lokasi?.nama ?? "Pendeteksi Cuaca"}</p>
        <IkonUtama className="ikon-utama" size={96} strokeWidth={1.25} aria-hidden />
        {gabungan && hujan ? (
          <>
            <p className="suhu">{Math.round(hujan.suhu)}°</p>
            <h1 className="kondisi">{gabungan.kesimpulan}</h1>
            <p className="sub">dalam 12 jam ke depan</p>
            <div className="chips">
              <span><Droplets size={16} aria-hidden /> Hujan {persen(gabungan.hujan)}</span>
              <span><Sun size={16} aria-hidden /> Cerah {persen(gabungan.cerah)}</span>
            </div>
            <p className="sepakat">
              {gabungan.sepakat
                ? <><CircleCheck size={16} aria-hidden /> Semua sumber sepakat</>
                : <><TriangleAlert size={16} aria-hidden /> Sumber berbeda pendapat, kurang pasti</>}
            </p>
          </>
        ) : status ? (
          <p className="sub" role="status"><LoaderCircle className="putar" size={18} aria-hidden /> {status}</p>
        ) : (
          <>
            <h1 className="kondisi">Bakal hujan atau cerah?</h1>
            <p className="sub">Pilih lokasi untuk melihat prediksi 12 jam ke depan.</p>
          </>
        )}
      </header>

      {error && <p className="error" role="alert"><TriangleAlert size={18} aria-hidden /> {error}</p>}
      {status && gabungan && <p className="sub" role="status"><LoaderCircle className="putar" size={18} aria-hidden /> {status}</p>}

      {hujan && (
        <section className="kartu" aria-labelledby="judul-jam">
          <h2 id="judul-jam"><Clock size={16} aria-hidden /> Per jam</h2>
          <div className="perjam">
            {hujan.perJam.map((p) => {
              const Ikon = ikonJam(p);
              return (
                <div key={p.jam} className="jam">
                  <span className="muted">{String(p.jam).padStart(2, "0")}.00</span>
                  <Ikon size={26} strokeWidth={1.5} aria-hidden />
                  <strong>{Math.round(p.suhu)}°</strong>
                  <span className={p.peluang >= 0.3 ? "peluang" : "peluang rendah"}>{persen(p.peluang)}</span>
                </div>
              );
            })}
          </div>
        </section>
      )}

      <section className="kartu" aria-labelledby="judul-foto">
        <h2 id="judul-foto"><Camera size={16} aria-hidden /> Foto langit <span className="muted">(opsional)</span></h2>
        {foto && fotoUrut && fotoAtas ? (
          <div className="foto">
            <img src={foto.url} alt="Foto langit yang dianalisis" />
            <div>
              <p className="foto-hasil"><fotoAtas.ikon size={20} aria-hidden /> {fotoAtas.nama}</p>
              {fotoUrut.slice(0, 3).map(([k, v]) => <Bar key={k} label={KELAS[k]?.nama ?? k} nilai={v} />)}
            </div>
          </div>
        ) : (
          <p className="muted">Foto cuma menggambarkan kondisi sekarang, jadi bobotnya kecil di hasil akhir.</p>
        )}
        <label className="tombol">
          <Camera size={18} aria-hidden /> {foto ? "Ganti foto" : "Ambil atau pilih foto"}
          <input type="file" accept="image/*" onChange={pilihFoto} />
        </label>
      </section>

      {hujan && (
        <section className="kartu" aria-labelledby="judul-sumber">
          <h2 id="judul-sumber">Rincian sumber</h2>
          <div className="sumber">
            <p><BrainCircuit size={18} aria-hidden /> <strong>Model LSTM</strong> <span className="muted">bobot 40%</span></p>
            <Bar label="Hujan" nilai={hujan.hujan} />
            <Bar label="Cerah" nilai={hujan.cerah} />
          </div>
          <div className="sumber">
            <p><Satellite size={18} aria-hidden /> <strong>Prakiraan Open-Meteo</strong> <span className="muted">bobot 40%</span></p>
            <Bar label="Hujan" nilai={hujan.openMeteo.peluangHujan} />
            <p className="muted">Total {hujan.openMeteo.hujanMm.toFixed(1)} mm · rata-rata awan {Math.round(hujan.openMeteo.awan)}%</p>
          </div>
          <div className="sumber">
            <p><Camera size={18} aria-hidden /> <strong>Foto (MobileNetV2)</strong> <span className="muted">bobot 20%</span></p>
            <p className="muted">{fotoAtas ? `${fotoAtas.nama} ${persen(fotoUrut![0][1])}` : "Belum ada foto"}</p>
          </div>
        </section>
      )}

      <footer className="muted">
        Model hujan (LSTM) dilatih dengan data Jakarta–Bekasi; di kota lain hasilnya kurang bisa dipercaya.
        Data cuaca: Open-Meteo.
      </footer>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
