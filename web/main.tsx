import { StrictMode, useEffect, useState, type ChangeEvent } from "react";
import { createRoot } from "react-dom/client";
import { ambilCuaca, gabungkan, prediksiHujan, type Config } from "./cuaca.ts";
import type { BobotLSTM } from "./lstm.ts";
import "./style.css";

const NAMA_KELAS: Record<string, string> = {
  cloudy: "Berawan", fogsmog: "Kabut/asap", lightning: "Petir", rain: "Hujan",
  rainbow: "Pelangi", shine: "Cerah", sunrise: "Matahari terbit/terbenam",
};
const KOTA = { Jakarta: [-6.2, 106.85], Bekasi: [-6.24, 106.99] } as const;
const persen = (v: number) => `${Math.round(v * 100)}%`;
const json = (p: string) => fetch(p).then((r) => r.json());

// Model hujan kecil (±400 KB) langsung dimuat; model foto (±9 MB) baru dimuat waktu ada foto
const modelHujan = Promise.all([json("/model/hujan_config.json"), json("/model/lstm_bobot.json")]) as Promise<[Config, BobotLSTM]>;
let modelFoto: Promise<[typeof import("./foto.ts"), Awaited<ReturnType<typeof import("./foto.ts").muatModelFoto>>]> | undefined;

function Bar({ label, nilai, warna }: { label: string; nilai: number; warna: string }) {
  return (
    <div className="bar">
      <span>{label}</span>
      <div className="isi"><div style={{ width: persen(nilai), background: warna }} /></div>
      <span>{persen(nilai)}</span>
    </div>
  );
}

function App() {
  const [lokasi, setLokasi] = useState<{ nama: string; lat: number; lon: number } | null>(null);
  const [hujan, setHujan] = useState<ReturnType<typeof prediksiHujan> | null>(null);
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

  const gabungan = hujan && gabungkan(hujan, foto?.hasil);
  const fotoUrut = foto && Object.entries(foto.hasil).sort((a, b) => b[1] - a[1]);

  return (
    <>
      <h1>Pendeteksi Cuaca</h1>
      <p className="muted" style={{ margin: 0 }}>Bakal hujan atau cerah dalam 12 jam ke depan?</p>

      <h2>1. Lokasi</h2>
      <div className="row">
        <button onClick={pakaiGPS}>📍 Pakai lokasi saya</button>
        {Object.entries(KOTA).map(([nama, [lat, lon]]) => (
          <button key={nama} className="ghost" onClick={() => setLokasi({ nama, lat, lon })}>{nama}</button>
        ))}
      </div>

      <h2>2. Foto langit (opsional)</h2>
      <label className="tombol ghost">
        📷 Pilih / ambil foto
        <input type="file" accept="image/*" onChange={pilihFoto} />
      </label>

      {status && <p className="muted">{status}</p>}
      {error && <p className="error">{error}</p>}

      {gabungan && hujan && (
        <div className="card" aria-live="polite">
          <p className="muted" style={{ margin: 0 }}>{lokasi?.nama} · 12 jam ke depan</p>
          <p className="kesimpulan">{gabungan.kesimpulan}</p>
          <Bar label="Hujan" nilai={gabungan.hujan} warna="var(--hujan)" />
          <Bar label="Cerah" nilai={gabungan.cerah} warna="var(--cerah)" />
          <p className="muted" style={{ marginBottom: 0 }}>
            {gabungan.sepakat ? "✅ Sumber-sumber sepakat." : "⚠️ Sumber berbeda pendapat, anggap kurang pasti."}
          </p>
        </div>
      )}

      {hujan && (
        <div className="card">
          <strong>Rincian sumber</strong>
          <table><tbody>
            <tr><td>LSTM (model kita): hujan</td><td>{persen(hujan.hujan)}</td></tr>
            <tr><td>LSTM (model kita): cerah</td><td>{persen(hujan.cerah)}</td></tr>
            <tr><td>Open-Meteo: peluang hujan</td><td>{persen(hujan.openMeteo.peluangHujan)}</td></tr>
            <tr><td>Open-Meteo: total hujan</td><td>{hujan.openMeteo.hujanMm.toFixed(1)} mm</td></tr>
            <tr><td>Open-Meteo: rata-rata awan</td><td>{Math.round(hujan.openMeteo.awan)}%</td></tr>
          </tbody></table>
        </div>
      )}

      {foto && fotoUrut && (
        <div className="card foto">
          <img src={foto.url} alt="Foto langit yang dipilih" />
          <div style={{ flex: 1 }}>
            <strong>Foto: {NAMA_KELAS[fotoUrut[0][0]] ?? fotoUrut[0][0]}</strong>
            {fotoUrut.slice(0, 3).map(([k, v]) => (
              <Bar key={k} label={NAMA_KELAS[k] ?? k} nilai={v} warna="var(--accent)" />
            ))}
          </div>
        </div>
      )}

      <p className="muted" style={{ marginTop: 24 }}>
        Model hujan (LSTM) dilatih dengan data Jakarta–Bekasi; di kota lain hasilnya kurang bisa dipercaya.
        Foto cuma menggambarkan kondisi <em>sekarang</em>, jadi bobotnya kecil. Data: Open-Meteo.
      </p>
    </>
  );
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
