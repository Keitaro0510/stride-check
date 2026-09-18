/* ステップ6：画面の計算。app_reference.py（Python の基準の実装）と同じ結果を出す。
 * ブラウザでは window.StrideModel、Node では module.exports で使う。
 * evaluate(measures, APP_DATA) → { speed, rhythm_type, legs, loads, flags }
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.StrideModel = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const TARGET_KMH = 10.0;
  const ANGLES = ["CPD", "HADD", "KF", "KA"];
  const LEG_LOAD_KEYS = ["CPD", "HADD", "KA", "KF", "knee_flex_contact", "shank_angle_contact", "foot_angle_contact", "overstride"];

  function num(x) {
    return typeof x === "number" && isFinite(x) ? x : null;
  }

  function speedInfo(m, data) {
    let s = num(m.subject && m.subject.speed_kmh);
    const assumed = s === null;
    if (assumed) s = TARGET_KMH;
    const [lo, hi] = data.speed_range_kmh;
    return { speed_kmh: s, assumed, delta_kmh: s - TARGET_KMH, out_of_range: !(lo <= s && s <= hi) };
  }

  function rhythmType(m, data, sp) {
    const t = data.types.rhythm;
    const cad = num(m.rhythm && m.rhythm.cadence_spm);
    const fs = m.rhythm && m.rhythm.foot_strike;
    if (cad === null || (fs !== "rearfoot" && fs !== "non_rearfoot")) return null;
    const cad10 = cad - t.speed_adjust_per_kmh.cadence_10 * sp.delta_kmh;
    const [lo, hi] = t.axes.x.threshold_ci95;
    const label = 2 * (cad10 > t.axes.x.threshold ? 1 : 0) + (fs === "rearfoot" ? 1 : 0);
    return { label, name: t.types[label].name, share: t.types[label].share, cadence_10: cad10,
             near_boundary: { cadence: lo <= cad10 && cad10 <= hi } };
  }

  // KF は動画からは測れないので（browser_pose_pipeline/validation/README.md）、
  // 無いときは研究データの中央値で補う。タイプの KF のしきい値がその中央値
  // （時速10kmでの重み付き中央値。step3_types/33_export_display_types.py）。
  function formValues10(leg, data, sp) {
    const adj = data.form_prediction.speed_adjust_per_kmh;
    const out = {}, imputed = [];
    for (const a of ANGLES) {
      const v = num(leg[a]);
      out[a] = v === null ? null : v - adj[a] * sp.delta_kmh;
    }
    if (out.KF === null) { out.KF = data.types.form.axes.y.threshold; imputed.push("KF"); }
    return { v10: out, imputed };
  }

  function formType(v10, data) {
    const t = data.types.form;
    if (v10.CPD === null || v10.KF === null) return null;
    const x = t.axes.x, y = t.axes.y;
    const label = 2 * (v10.CPD > x.threshold ? 1 : 0) + (v10.KF > y.threshold ? 1 : 0);
    return { label, name: t.types[label].name, share: t.types[label].share,
             near_boundary: { CPD: x.threshold_ci95[0] <= v10.CPD && v10.CPD <= x.threshold_ci95[1],
                              KF: y.threshold_ci95[0] <= v10.KF && v10.KF <= y.threshold_ci95[1] } };
  }

  function logistic(p, x) {
    let z = p.intercept;
    for (let i = 0; i < x.length; i++) z += p.coef[i] * (x[i] - p.scaler_mean[i]) / p.scaler_sd[i];
    return 1 / (1 + Math.exp(-z));
  }

  function percentile(values, q) {
    const v = values.slice().sort((a, b) => a - b);
    const pos = (v.length - 1) * q / 100;
    const lo = Math.floor(pos), hi = Math.min(lo + 1, v.length - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
  }

  function formPrediction(v10, data) {
    const fp = data.form_prediction;
    const x = fp.features.map((a) => v10[a]);
    if (x.some((v) => v === null || v === undefined)) return null;
    const boots = fp.bootstrap_params.map((b) => logistic(b, x));
    return { risk: logistic(fp.params, x), lo: percentile(boots, 2.5), hi: percentile(boots, 97.5) };
  }

  function ridge(model, inputs) {
    let total = model.intercept;
    model.features.forEach((f, i) => {
      let v = num(inputs[f]);
      if (v === null) v = model.impute_median[i];
      total += model.coef[i] * (v - model.scaler_mean[i]) / model.scaler_scale[i];
    });
    return total;
  }

  function loadInputs(m, leg, data) {
    const subj = m.subject || {}, rh = m.rhythm || {}, def = data.load_defaults;
    const speed = num(subj.speed_kmh), mass = num(subj.mass_kg), height = num(subj.height_cm);
    const out = {
      speed_ms: (speed === null ? TARGET_KMH : speed) / 3.6,
      mass_kg: mass === null ? def.mass_kg : mass,
      height_cm: height === null ? def.height_cm : height,
      cadence_spm: rh.cadence_spm, duty: rh.duty, contact_s: rh.contact_s,
      pelvis_vertical_osc: rh.pelvis_vertical_osc_mm,
    };
    for (const k of LEG_LOAD_KEYS) out[k] = leg[k];
    return out;
  }

  function presentLegs(m) {
    return ["L", "R"].filter((s) => m.legs && m.legs[s] && Object.keys(m.legs[s]).length > 0);
  }

  function loads(m, data) {
    const out = {};
    const legs = presentLegs(m);
    for (const [key, r] of Object.entries(data.loads)) {
      if (!r.individual_difference_estimable) { out[key] = { estimable: false, stars: r.stars }; continue; }
      const preds = legs.map((s) => ridge(r.model, loadInputs(m, m.legs[s], data)));
      const value = preds.reduce((a, b) => a + b, 0) / preds.length;
      out[key] = { estimable: true, value, pct: (value - r.reference_mean) / r.reference_mean * 100, error_pct: r.error_pct,
                   stars: r.stars, cadence_sim: r.show_cadence_sim ? r.cadence_sim : null };
    }
    return out;
  }

  function flags(m, data) {
    const out = [], dist = data.plausibility, rh = m.rhythm || {};
    for (const k of ["cadence_spm", "duty", "contact_s", "flight_s", "pelvis_vertical_osc_mm"]) {
      const v = num(rh[k]);
      if (v !== null && dist[k] && Math.abs(v - dist[k].mean) > 3 * dist[k].sd) out.push({ where: "rhythm", item: k, value: v });
    }
    for (const [s, leg] of Object.entries(m.legs || {})) {
      for (const [k, raw] of Object.entries(leg || {})) {
        const v = num(raw);
        if (v !== null && dist[k] && Math.abs(v - dist[k].mean) > 3 * dist[k].sd) out.push({ where: s, item: k, value: v });
      }
    }
    return out;
  }

  function evaluate(m, data) {
    const sp = speedInfo(m, data);
    const legs = {};
    for (const [s, leg] of Object.entries(m.legs || {})) {
      if (!leg || Object.keys(leg).length === 0) continue;
      const { v10, imputed } = formValues10(leg, data, sp);
      legs[s] = { values_10: v10, imputed, type: formType(v10, data), prediction: formPrediction(v10, data) };
    }
    return { speed: sp, rhythm_type: rhythmType(m, data, sp), legs, loads: loads(m, data), flags: flags(m, data) };
  }

  return { evaluate, TARGET_KMH };
});
