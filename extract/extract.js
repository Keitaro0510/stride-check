/*
 * Browser-only video -> measures.json adapter for Stride Check.
 *
 * The video never leaves the device.  Runtime/model assets are loaded from
 * the URLs below unless configure({ assetBaseUrl }) points at self-hosted
 * copies.  It is deliberately a single classic script because index.html
 * already loads extract/extract.js that way.
 */
(function (root) {
  "use strict";

  const VERSION = "mediapipe-pose-browser-0.1.0";
  const MP_VERSION = "0.10.22";
  const DEFAULT_ASSET_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@" + MP_VERSION;
  const DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";
  const ANALYSIS_FPS = 60;
  const MIN_STEPS = 7;
  const MAX_SECONDS = 20;
  const LANDMARK = { NOSE: 0, LEFT_HIP: 23, RIGHT_HIP: 24, LEFT_KNEE: 25,
    RIGHT_KNEE: 26, LEFT_ANKLE: 27, RIGHT_ANKLE: 28, LEFT_HEEL: 29,
    RIGHT_HEEL: 30, LEFT_TOE: 31, RIGHT_TOE: 32 };
  let config = { assetBaseUrl: DEFAULT_ASSET_BASE, modelUrl: DEFAULT_MODEL_URL };
  let landmarkerPromise = null;

  function finite(v) { return Number.isFinite(v); }
  function numeric(value) {
    const n = typeof value === "number" ? value : (typeof value === "string" && value.trim() ? Number(value) : NaN);
    return finite(n) ? n : null;
  }
  function round(v, digits) { return finite(v) ? Number(v.toFixed(digits == null ? 4 : digits)) : null; }
  function median(values) {
    const a = values.filter(finite).sort((x, y) => x - y);
    if (!a.length) return null;
    const m = Math.floor(a.length / 2);
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
  }
  function sd(values) {
    const a = values.filter(finite);
    if (a.length < 2) return null;
    const mean = a.reduce((s, v) => s + v, 0) / a.length;
    return Math.sqrt(a.reduce((s, v) => s + (v - mean) ** 2, 0) / (a.length - 1));
  }
  function point(frame, index) { return frame && frame[index] ? frame[index] : null; }
  function ok(p) { return p && finite(p.x) && finite(p.y) && (p.visibility == null || p.visibility >= 0.25); }
  function dist(a, b) { return ok(a) && ok(b) ? Math.hypot(a.x - b.x, a.y - b.y) : NaN; }
  function angle(a, b, c) {
    if (!ok(a) || !ok(b) || !ok(c)) return NaN;
    const ux = a.x - b.x, uy = a.y - b.y, vx = c.x - b.x, vy = c.y - b.y;
    const d = Math.hypot(ux, uy) * Math.hypot(vx, vy);
    return d ? Math.acos(Math.max(-1, Math.min(1, (ux * vx + uy * vy) / d))) * 180 / Math.PI : NaN;
  }
  function vectorAngle(a, b) {
    if (!finite(a.x) || !finite(a.y) || !finite(b.x) || !finite(b.y)) return NaN;
    const d = Math.hypot(a.x, a.y) * Math.hypot(b.x, b.y);
    return d ? Math.acos(Math.max(-1, Math.min(1, (a.x * b.x + a.y * b.y) / d))) * 180 / Math.PI : NaN;
  }
  function lerpFrame(frames, fraction) {
    if (!frames.length) return null;
    return frames[Math.max(0, Math.min(frames.length - 1, Math.round(fraction * (frames.length - 1))))];
  }
  function movingAverage(values, radius) {
    return values.map((_, i) => {
      const part = values.slice(Math.max(0, i - radius), Math.min(values.length, i + radius + 1)).filter(finite);
      return part.length ? part.reduce((s, v) => s + v, 0) / part.length : NaN;
    });
  }
  function localMaxima(values, minDistance, threshold) {
    const found = [];
    for (let i = 1; i < values.length - 1; i++) {
      if (!finite(values[i]) || values[i] < threshold || values[i] < values[i - 1] || values[i] < values[i + 1]) continue;
      if (found.length && i - found[found.length - 1] < minDistance) {
        if (values[i] > values[found[found.length - 1]]) found[found.length - 1] = i;
      } else found.push(i);
    }
    return found;
  }
  function progressionSign(frames) {
    const values = frames.map(f => {
      const nose = point(f, LANDMARK.NOSE), l = point(f, LANDMARK.LEFT_HIP), r = point(f, LANDMARK.RIGHT_HIP);
      return ok(nose) && ok(l) && ok(r) ? nose.x - (l.x + r.x) / 2 : NaN;
    });
    return (median(values) || 0) >= 0 ? 1 : -1;
  }
  function contactsFor(frames, leg, fps, direction) {
    const heel = leg === "L" ? LANDMARK.LEFT_HEEL : LANDMARK.RIGHT_HEEL;
    const raw = frames.map(f => { const p = point(f, heel); return ok(p) ? direction * p.x : NaN; });
    const smoothed = movingAverage(raw, Math.max(1, Math.round(fps * 0.04)));
    const valid = smoothed.filter(finite);
    if (!valid.length) return [];
    const lo = Math.min(...valid), hi = Math.max(...valid);
    return localMaxima(smoothed, Math.max(3, Math.round(fps * 0.45)), lo + (hi - lo) * 0.50);
  }
  function toeOff(frames, leg, contact, nextContact) {
    const toe = leg === "L" ? LANDMARK.LEFT_TOE : LANDMARK.RIGHT_TOE;
    const ys = frames.slice(contact, nextContact).map(f => { const p = point(f, toe); return ok(p) ? p.y : NaN; });
    const valid = ys.filter(finite);
    if (valid.length < 4) return null;
    // In image coordinates a larger y is lower.  The toe reaches its lowest
    // point in late stance; detect the following lift through 68% of its span.
    const low = Math.min(...valid), high = Math.max(...valid), threshold = low + (high - low) * 0.68;
    let peak = ys.reduce((best, value, i) => finite(value) && value > ys[best] ? i : best, 0);
    for (let i = peak + 1; i < ys.length; i++) if (finite(ys[i]) && ys[i] < threshold) return contact + i;
    return null;
  }
  function sideValues(frame, leg, direction) {
    const hip = point(frame, leg === "L" ? LANDMARK.LEFT_HIP : LANDMARK.RIGHT_HIP);
    const knee = point(frame, leg === "L" ? LANDMARK.LEFT_KNEE : LANDMARK.RIGHT_KNEE);
    const ankle = point(frame, leg === "L" ? LANDMARK.LEFT_ANKLE : LANDMARK.RIGHT_ANKLE);
    const heel = point(frame, leg === "L" ? LANDMARK.LEFT_HEEL : LANDMARK.RIGHT_HEEL);
    const toe = point(frame, leg === "L" ? LANDMARK.LEFT_TOE : LANDMARK.RIGHT_TOE);
    const lh = point(frame, LANDMARK.LEFT_HIP), rh = point(frame, LANDMARK.RIGHT_HIP);
    const pelvis = ok(lh) && ok(rh) ? { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 } : null;
    const legLength = dist(hip, ankle);
    return {
      KF: 180 - angle(hip, knee, ankle),
      knee_flex_contact: 180 - angle(hip, knee, ankle),
      shank_angle_contact: ok(knee) && ok(ankle) ? Math.atan2(direction * (knee.x - ankle.x), ankle.y - knee.y) * 180 / Math.PI : NaN,
      foot_angle_contact: ok(heel) && ok(toe) ? Math.atan2(heel.y - toe.y, direction * (toe.x - heel.x)) * 180 / Math.PI : NaN,
      overstride: ok(heel) && pelvis && finite(legLength) && legLength > 0 ? direction * (heel.x - pelvis.x) / legLength : NaN,
    };
  }
  function rearValues(frame, leg) {
    const other = leg === "L" ? "R" : "L";
    const hip = point(frame, leg === "L" ? LANDMARK.LEFT_HIP : LANDMARK.RIGHT_HIP);
    const contra = point(frame, other === "L" ? LANDMARK.LEFT_HIP : LANDMARK.RIGHT_HIP);
    const knee = point(frame, leg === "L" ? LANDMARK.LEFT_KNEE : LANDMARK.RIGHT_KNEE);
    const ankle = point(frame, leg === "L" ? LANDMARK.LEFT_ANKLE : LANDMARK.RIGHT_ANKLE);
    if (!ok(hip) || !ok(contra) || !ok(knee) || !ok(ankle)) return { CPD: NaN, HADD: NaN, KA: NaN };
    const cpd = Math.atan2(contra.y - hip.y, Math.abs(contra.x - hip.x)) * 180 / Math.PI;
    const hadd = vectorAngle({ x: contra.x - hip.x, y: contra.y - hip.y }, { x: knee.x - hip.x, y: knee.y - hip.y });
    const magnitude = 180 - angle(hip, knee, ankle);
    const fraction = ankle.y === hip.y ? .5 : (knee.y - hip.y) / (ankle.y - hip.y);
    const lineX = hip.x + fraction * (ankle.x - hip.x);
    const outside = (leg === "L" ? -1 : 1) * (knee.x - lineX);
    return { CPD: cpd, HADD: hadd, KA: Math.sign(outside || 1) * magnitude };
  }
  function emptyLeg() {
    return { CPD: null, HADD: null, KA: null, KF: null, knee_flex_contact: null,
      shank_angle_contact: null, foot_angle_contact: null, overstride: null, n_steps: 0 };
  }
  async function importVision() {
    // Bundlers cannot rewrite a runtime URL.  Keeping it explicit also makes
    // self-hosting possible through configure().
    return import(config.assetBaseUrl + "/+esm");
  }
  async function getLandmarker() {
    if (!landmarkerPromise) landmarkerPromise = (async () => {
      const vision = await importVision();
      const fileset = await vision.FilesetResolver.forVisionTasks(config.assetBaseUrl + "/wasm");
      const options = delegate => ({
        baseOptions: { modelAssetPath: config.modelUrl, delegate }, runningMode: "VIDEO",
        numPoses: 1, minPoseDetectionConfidence: .55, minPosePresenceConfidence: .55,
        minTrackingConfidence: .55,
      });
      try { return await vision.PoseLandmarker.createFromOptions(fileset, options("GPU")); }
      catch (_) { return vision.PoseLandmarker.createFromOptions(fileset, options("CPU")); }
    })();
    return landmarkerPromise;
  }
  function waitEvent(target, type) { return new Promise((resolve, reject) => {
    const okHandler = () => { cleanup(); resolve(); }, fail = () => { cleanup(); reject(new Error("Could not read the selected video.")); };
    const cleanup = () => { target.removeEventListener(type, okHandler); target.removeEventListener("error", fail); };
    target.addEventListener(type, okHandler, { once: true }); target.addEventListener("error", fail, { once: true });
  }); }
  async function openVideo(file) {
    const video = document.createElement("video");
    video.muted = true; video.playsInline = true; video.preload = "auto";
    const url = URL.createObjectURL(file); video.src = url;
    await waitEvent(video, "loadedmetadata");
    if (!finite(video.duration) || video.duration <= 0) { URL.revokeObjectURL(url); throw new Error("The video duration is unavailable."); }
    return { video, url };
  }
  async function seek(video, seconds) {
    if (Math.abs(video.currentTime - seconds) < .0001) return;
    const loaded = waitEvent(video, "seeked"); video.currentTime = seconds; await loaded;
  }
  async function inferVideo(file, targetFps, progress, progressBase, progressSpan, signal) {
    const { video, url } = await openVideo(file);
    try {
      const landmarker = await getLandmarker();
      const duration = Math.min(video.duration, MAX_SECONDS);
      const count = Math.max(2, Math.ceil(duration * targetFps));
      const frames = [];
      for (let i = 0; i < count; i++) {
        if (signal && signal.aborted) throw new DOMException("Analysis cancelled", "AbortError");
        const seconds = Math.min(duration - .001, i / targetFps);
        await seek(video, seconds);
        const result = landmarker.detectForVideo(video, Math.round(seconds * 1000));
        frames.push(result.landmarks && result.landmarks[0] ? result.landmarks[0] : null);
        if (i % 3 === 0 || i === count - 1) progress(progressBase + progressSpan * (i + 1) / count, "Estimating pose");
      }
      return { frames, duration, sampled: count, originalDuration: video.duration };
    } finally { URL.revokeObjectURL(url); }
  }
  function summarize(sideFrames, rearFrames, fps, heightCm, rearProvided, notes) {
    const direction = progressionSign(sideFrames);
    const contacts = { L: contactsFor(sideFrames, "L", fps, direction), R: contactsFor(sideFrames, "R", fps, direction) };
    const allContacts = Object.entries(contacts).flatMap(([leg, xs]) => xs.map(frame => ({ frame, leg }))).sort((a, b) => a.frame - b.frame);
    const values = { L: [], R: [] }, steps = { L: [], R: [] }, duties = { L: [], R: [] }, pelvisOsc = [];
    const headToFoot = sideFrames.map(f => {
      const nose = point(f, LANDMARK.NOSE), lh = point(f, LANDMARK.LEFT_HEEL), rh = point(f, LANDMARK.RIGHT_HEEL);
      return ok(nose) && (ok(lh) || ok(rh)) ? Math.max((lh || rh).y, (rh || lh).y) - nose.y : NaN;
    });
    const heightPx = median(headToFoot);
    const mmPerUnit = finite(heightCm) && finite(heightPx) && heightPx > 0 ? heightCm * 10 / heightPx : NaN;
    for (const leg of ["L", "R"]) for (let i = 0; i + 1 < contacts[leg].length; i++) {
      const contact = contacts[leg][i], next = contacts[leg][i + 1];
      const opposite = allContacts.find(e => e.frame > contact && e.leg !== leg);
      if (!opposite || opposite.frame >= next) continue;
      const off = toeOff(sideFrames, leg, contact, next);
      if (off == null || off <= contact || off >= next) continue;
      const mid = Math.round((contact + off) / 2), stepS = (opposite.frame - contact) / fps, contactS = (off - contact) / fps;
      const sideMid = sideValues(sideFrames[mid], leg, direction), sideContact = sideValues(sideFrames[contact], leg, direction);
      const rear = rearProvided ? rearValues(lerpFrame(rearFrames, mid / Math.max(1, sideFrames.length - 1)), leg) : { CPD: NaN, HADD: NaN, KA: NaN };
      const interval = sideFrames.slice(contact, opposite.frame + 1).map(f => {
        const l = point(f, LANDMARK.LEFT_HIP), r = point(f, LANDMARK.RIGHT_HIP); return ok(l) && ok(r) ? (l.y + r.y) / 2 : NaN;
      }).filter(finite);
      if (interval.length && finite(mmPerUnit)) pelvisOsc.push((Math.max(...interval) - Math.min(...interval)) * mmPerUnit);
      const row = { ...rear, KF: sideMid.KF, knee_flex_contact: sideContact.knee_flex_contact,
        shank_angle_contact: sideContact.shank_angle_contact, foot_angle_contact: sideContact.foot_angle_contact,
        overstride: sideContact.overstride, step_s: stepS, contact_s: contactS,
        flight_s: Math.max(0, stepS - contactS), duty: stepS > 0 ? contactS / stepS : NaN };
      values[leg].push(row); steps[leg].push(stepS); duties[leg].push(row.duty);
    }
    const metricNames = ["CPD", "HADD", "KA", "KF", "knee_flex_contact", "shank_angle_contact", "foot_angle_contact", "overstride"];
    const legs = {};
    for (const leg of ["L", "R"]) {
      legs[leg] = emptyLeg(); legs[leg].n_steps = values[leg].length;
      metricNames.forEach(k => { legs[leg][k] = round(median(values[leg].map(v => v[k]))); });
    }
    const all = values.L.concat(values.R);
    const strike = leg => { const a = median(values[leg].map(v => v.foot_angle_contact)); return finite(a) ? (a > 5 ? "rearfoot" : "non_rearfoot") : null; };
    const lStrike = strike("L"), rStrike = strike("R");
    function asym(a, b) { const x = median(a), y = median(b); return finite(x) && finite(y) && x + y ? Math.abs(x - y) / ((x + y) / 2) * 100 : null; }
    if (!rearProvided) notes.push("No rear video was supplied; CPD, HADD, and KA are unavailable.");
    if (Math.min(values.L.length, values.R.length) < MIN_STEPS) notes.push("Fewer than seven valid steps per leg were measured; record a longer, clearer trial.");
    if (!finite(heightPx) || heightPx < .5) notes.push("Full-body height could not be estimated reliably; pelvis vertical oscillation may be unavailable.");
    return {
      rhythm: { cadence_spm: round((() => { const v = median(all.map(s => s.step_s)); return finite(v) && v > 0 ? 60 / v : NaN; })()),
        contact_s: round(median(all.map(s => s.contact_s))), flight_s: round(median(all.map(s => s.flight_s))), duty: round(median(all.map(s => s.duty))),
        foot_strike: lStrike === rStrike ? lStrike : null, cadence_asym_pct: round(asym(steps.L, steps.R)), duty_asym_pct: round(asym(duties.L, duties.R)),
        alt_strike: lStrike && rStrike ? lStrike !== rStrike : null, pelvis_vertical_osc_mm: round(median(pelvisOsc)) },
      legs, quality: { step_sd: Object.fromEntries(metricNames.concat(["contact_s", "flight_s", "duty"]).map(k => [k, round(sd(all.map(v => v[k])))])),
        low_confidence_frames_pct: null, notes }
    };
  }
  async function extractMeasures(files, subject, options) {
    const rearFile = files && files.rear, sideFile = files && files.side;
    if (!sideFile) throw new Error("A side-view video is required to measure timing and running form.");
    options = options || {}; const progress = typeof options.onProgress === "function" ? options.onProgress : () => {};
    const fps = finite(options.fps) && options.fps >= 30 ? options.fps : ANALYSIS_FPS;
    const notes = [];
    if (!finite(options.fps)) notes.push("Source frame rate is unavailable in browser media metadata; values were sampled at 60 Hz and timing needs validation against the original capture rate.");
    progress(.01, "Loading pose model"); await getLandmarker();
    const side = await inferVideo(sideFile, fps, progress, .03, rearFile ? .48 : .94, options.signal);
    const rear = rearFile ? await inferVideo(rearFile, fps, progress, .51, .43, options.signal) : { frames: [] };
    if (side.originalDuration > MAX_SECONDS || (rearFile && rear.originalDuration > MAX_SECONDS)) {
      notes.push("Only the first 20 seconds of each video were analysed; trim the recording to a steady running section.");
    }
    if (rearFile) notes.push("Rear and side videos were aligned by their relative duration. Use synchronized recordings for valid frontal-plane angles.");
    progress(.96, "Calculating running measures");
    const measured = summarize(side.frames, rear.frames, fps, numeric(subject && subject.height_cm), Boolean(rearFile), notes);
    progress(1, "Done");
    return { video: { rear: rearFile ? rearFile.name : null, side: sideFile.name, fps_rear: rearFile ? fps : null, fps_side: fps,
        fps_source: finite(options.fps) ? "user" : "estimated" },
      subject: { speed_kmh: numeric(subject && subject.speed_kmh), height_cm: numeric(subject && subject.height_cm),
        mass_kg: numeric(subject && subject.mass_kg), sex: subject && subject.sex || null },
      ...measured, model: { pose: "MediaPipe Pose Landmarker full (" + MP_VERSION + ")", extractor: VERSION } };
  }
  root.StrideExtract = { available: true, version: VERSION,
    configure(next) { config = Object.assign({}, config, next || {}); landmarkerPromise = null; }, extractMeasures };
})(typeof self !== "undefined" ? self : this);
