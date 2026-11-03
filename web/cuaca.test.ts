import { test } from "node:test";
import assert from "node:assert/strict";
import { fitur } from "./cuaca.ts";

test("fitur sama dengan notebook (log1p hujan, jam & hari sin/cos)", () => {
  const jam = {
    time: ["2026-01-01T06:00"],
    temperature_2m: [25], relative_humidity_2m: [80], pressure_msl: [1010],
    cloud_cover: [50], wind_speed_10m: [5], precipitation: [Math.E - 1],
  };
  const f = fitur(jam, 0);
  assert.deepEqual(f.slice(0, 5), [25, 80, 1010, 50, 5]);
  assert.ok(Math.abs(f[5] - 1) < 1e-12);                                  // log1p(e-1) = 1
  assert.ok(Math.abs(f[6] - 1) < 1e-12 && Math.abs(f[7]) < 1e-12);         // jam 6 -> sin 1, cos 0
  assert.ok(Math.abs(f[8] - Math.sin((2 * Math.PI) / 365.25)) < 1e-12);    // 1 Januari = hari ke-1
});
