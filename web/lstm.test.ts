import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { jalankanLSTM } from "./lstm.ts";

const baca = (p: string) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));

test("jalankanLSTM sama dengan Keras", () => {
  const bobot = baca("../public/model/lstm_bobot.json");
  const { x, y } = baca("./lstm.fixture.json"); // dibuat ekspor_web.py
  const hasil = jalankanLSTM(bobot, x);
  hasil.forEach((v, i) => assert.ok(Math.abs(v - y[i]) < 1e-4, `output ${i}: ${v} vs keras ${y[i]}`));
});
