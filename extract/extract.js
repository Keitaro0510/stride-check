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

  const VERSION = "mediapipe-pose-browser-0.6.0";
  // 0.10.22 was never published to npm.  Pin the currently published
  // stable package so the CDN URL is a real, cacheable asset.
  const MP_VERSION = "1.0.1";
  const DEFAULT_ASSET_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@" + MP_VERSION;
  // Heavy (30.7 MB): against motion capture it measured contact time and the
  // foot angle at contact clearly better than Full (validation/README.md).
  const DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task";
  // Sampling rate used only when neither the file nor the user gives the
  // capture rate.  Timing values are then withheld (see summarize()).
  const FALLBACK_SAMPLING_FPS = 60;
  const MIN_STEPS = 7;
  // Event timing corrections (validation/README.md).  The heel's most forward
  // point comes 29 ms before initial contact (Fukuchi 2017 markers, 95
  // trials; unbiased on MediaPipe points in the Twente videos too).  The
  // toe's most rearward point relative to the pelvis comes 12.0% of the
  // stride after toe-off on MediaPipe Heavy's toe tip (Twente, 24 trials,
  // force plates, leave-one-subject-out; 11.3% with Full); on
  // metatarsal-head markers it is 16.6%.
  const IC_AFTER_HEEL_FORWARD_S = 0.029;
  const TO_BEFORE_TOE_BACK_STRIDE = 0.120;
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
  // Frame of `clip` shown at the centre of side frame `index` (both clips are
  // assumed to start at the same instant).
  function frameAt(clip, index, fromFps) {
    if (!clip.frames.length) return null;
    const j = Math.floor((index + .5) / fromFps * clip.fps);
    return clip.frames[Math.max(0, Math.min(clip.frames.length - 1, j))];
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
  // Running direction in the side view (+1 = toward larger x).  The toes
  // point forward from the heels; the nose (ahead of the hips) is only a
  // fallback, because the head is often cropped and its point then wanders.
  function progressionSign(frames) {
    const feet = frames.map(f => {
      const d = [[LANDMARK.LEFT_HEEL, LANDMARK.LEFT_TOE], [LANDMARK.RIGHT_HEEL, LANDMARK.RIGHT_TOE]]
        .map(([h, t]) => ok(point(f, h)) && ok(point(f, t)) ? f[t].x - f[h].x : NaN).filter(finite);
      return d.length ? d.reduce((a, b) => a + b, 0) / d.length : NaN;
    });
    const foot = median(feet);
    if (finite(foot) && foot !== 0) return foot > 0 ? 1 : -1;
    const values = frames.map(f => {
      const nose = point(f, LANDMARK.NOSE), l = point(f, LANDMARK.LEFT_HIP), r = point(f, LANDMARK.RIGHT_HIP);
      return ok(nose) && ok(l) && ok(r) ? nose.x - (l.x + r.x) / 2 : NaN;
    });
    return (median(values) || 0) >= 0 ? 1 : -1;
  }
  /* ---- Side view: keep each leg's identity through leg crossings ----
   * In a side view MediaPipe tends to label the legs by position (front =
   * one side, back = the other), so the labels swap at every crossing, and
   * while the legs cross the far leg's points are pulled onto the near leg.
   * 1. Frames where the two leg chains are close ("crossing") or missing are
   *    set aside.
   * 2. On the remaining frames, re-assign the two chains by motion continuity
   *    (second-order Viterbi: least change of velocity over the whole clip).
   * 3. Through each crossing, predict each leg's path from the frames on
   *    either side (cubic Hermite).  A detected point is kept when it lies on
   *    that path; otherwise the prediction is used (visibility 0.3). */
  const LEG_CHAIN = { L: [25, 27, 29, 31], R: [26, 28, 30, 32] };
  function cleanLegs(frames, fps, opts) {
    const clearSep = opts && opts.clearSep || .3;
    const out = frames.map(f => f ? f.slice() : null);
    const legKeys = LEG_CHAIN.L.concat(LEG_CHAIN.R);
    const has = f => f && legKeys.every(k => f[k] && finite(f[k].x) && finite(f[k].y));
    const info = { relabelled: 0, crossingFrames: 0, predicted: 0, naming: "labels" };
    const legLen = median(frames.map(f => has(f) ? dist(f[LANDMARK.LEFT_HIP], f[LANDMARK.LEFT_ANKLE]) : NaN));
    if (!finite(legLen)) return { frames: out, info };
    // Mean distance between the two chains' ankle, heel and toe.
    const sep = f => [1, 2, 3].reduce((s, j) => s + Math.hypot(f[LEG_CHAIN.L[j]].x - f[LEG_CHAIN.R[j]].x, f[LEG_CHAIN.L[j]].y - f[LEG_CHAIN.R[j]].y), 0) / 3;
    const clear = frames.map(f => has(f) && sep(f) >= legLen * clearSep);
    const idx = clear.map((c, i) => c ? i : -1).filter(i => i >= 0);
    if (idx.length < 3) return { frames: out, info };
    const chain = (i, s, t) => LEG_CHAIN[(t ^ s) ? "R" : "L"].map(k => frames[i][k]);
    // The legs cross once per step, and between two crossings the same leg
    // stays in front.  So find the crossings (the two feet closest along the
    // running direction), make them regular, and alternate which leg is in
    // front at each one.  Only the front/back order of the feet is used, so
    // this works whether the model labels legs by position or by side.
    const dirX = progressionSign(frames);
    const fwd = (f, leg) => [1, 2, 3].reduce((acc, j) => acc + dirX * f[LEG_CHAIN[leg][j]].x / 3, 0);
    const gapRaw = frames.map(f => has(f) ? Math.abs(fwd(f, "L") - fwd(f, "R")) : NaN);
    const gapS = movingAverage(gapRaw, 1);
    const minDist = Math.max(2, Math.round(fps * .2));
    let crossings = [];
    for (let i = 1; i + 1 < gapS.length; i++) {
      if (!finite(gapS[i]) || gapS[i] > legLen * .3) continue;
      const l = finite(gapS[i - 1]) ? gapS[i - 1] : Infinity, r = finite(gapS[i + 1]) ? gapS[i + 1] : Infinity;
      if (gapS[i] > l || gapS[i] > r) continue;
      const last = crossings[crossings.length - 1];
      if (last != null && i - last < minDist) { if (gapS[i] < gapS[last]) crossings[crossings.length - 1] = i; }
      else crossings.push(i);
    }
    // Regularise with the median interval: drop one of two crossings that are
    // too close, add one in the middle of an interval that is about double.
    const typical = median(crossings.slice(1).map((c, k) => c - crossings[k]));
    if (finite(typical)) {
      const kept = [];
      for (const c of crossings) {
        const last = kept[kept.length - 1];
        if (last != null && c - last < typical * .5) { if (gapS[c] < gapS[last]) kept[kept.length - 1] = c; }
        else kept.push(c);
      }
      crossings = [];
      kept.forEach((c, k) => {
        const prev = crossings[crossings.length - 1];
        if (prev != null) { const n = Math.round((c - prev) / typical); for (let m = 1; m < n; m++) crossings.push(Math.round(prev + (c - prev) * m / n)); }
        crossings.push(c);
      });
    }
    info.crossings = crossings.length;
    info.crossingAt = crossings.slice();
    // Track 0 is the leg in front before the first crossing.
    const state = idx.map(i => {
      const parity = crossings.filter(c => c <= i).length % 2;
      const frontIsL = fwd(frames[i], "L") >= fwd(frames[i], "R");
      // state 0 means track 0 = "L" label; track 0 is in front on even parity
      return (frontIsL === (parity === 0)) ? 0 : 1;
    });
    // Provisional names: the track carrying the "left" label more often.
    // Labels are position-based, so nameLegsByFrontal() settles it when a
    // front or rear video exists.  (MediaPipe depth follows the labels too.)
    const leftTrack = state.filter(v => v === 0).length >= state.length / 2 ? 0 : 1;
    const track = { L: new Array(frames.length).fill(null), R: new Array(frames.length).fill(null) };
    idx.forEach((i, k) => {
      if ((state[k] ^ leftTrack) !== 0) info.relabelled++;
      track.L[i] = chain(i, state[k], leftTrack); track.R[i] = chain(i, state[k], 1 - leftTrack);
    });
    // Crossings: between consecutive clear frames a and b, at most 0.3 s apart.
    const tol = legLen * .12, maxRun = Math.round(fps * .3);
    for (let k = 0; k + 1 < idx.length; k++) {
      const a = idx[k], b = idx[k + 1];
      if (b - a < 2 || b - a > maxRun) continue;
      for (const leg of ["L", "R"]) for (let j = 0; j < 4; j++) {
        const pa = track[leg][a][j], pb = track[leg][b][j];
        // Per-frame velocities at both ends (from the neighbouring clear frames).
        const prev = track[leg][a - 1], nxt = track[leg][b + 1];
        const va = prev ? { x: pa.x - prev[j].x, y: pa.y - prev[j].y } : { x: (pb.x - pa.x) / (b - a), y: (pb.y - pa.y) / (b - a) };
        const vb = nxt ? { x: nxt[j].x - pb.x, y: nxt[j].y - pb.y } : { x: (pb.x - pa.x) / (b - a), y: (pb.y - pa.y) / (b - a) };
        for (let m = a + 1; m < b; m++) {
          const u = (m - a) / (b - a), n = b - a;
          const h00 = 2 * u ** 3 - 3 * u ** 2 + 1, h10 = u ** 3 - 2 * u ** 2 + u, h01 = -2 * u ** 3 + 3 * u ** 2, h11 = u ** 3 - u ** 2;
          const pred = { x: h00 * pa.x + h10 * n * va.x + h01 * pb.x + h11 * n * vb.x, y: h00 * pa.y + h10 * n * va.y + h01 * pb.y + h11 * n * vb.y };
          const f = frames[m];
          const cands = f ? [f[LEG_CHAIN.L[j]], f[LEG_CHAIN.R[j]]].filter(p => p && finite(p.x) && finite(p.y)) : [];
          const near = cands.sort((p, q) => Math.hypot(p.x - pred.x, p.y - pred.y) - Math.hypot(q.x - pred.x, q.y - pred.y))[0];
          if (!track[leg][m]) track[leg][m] = [null, null, null, null];
          if (near && Math.hypot(near.x - pred.x, near.y - pred.y) <= tol) track[leg][m][j] = near;
          else { track[leg][m][j] = { x: pred.x, y: pred.y, visibility: .3 }; info.predicted++; }
        }
      }
      info.crossingFrames += b - a - 1;
    }
    for (let i = 0; i < frames.length; i++) {
      if (!track.L[i] && !track.R[i]) continue;   // long gaps stay as detected
      if (!out[i]) out[i] = [];
      LEG_CHAIN.L.forEach((k, j) => { out[i][k] = track.L[i] ? track.L[i][j] : null; });
      LEG_CHAIN.R.forEach((k, j) => { out[i][k] = track.R[i] ? track.R[i][j] : null; });
    }
    // Frames with no person detected: fill the other points (hips etc.) over
    // gaps of up to 0.3 s by linear interpolation.
    for (let k = 0; k < 33; k++) {
      if (legKeys.includes(k)) continue;
      const okAt = i => out[i] && out[i][k] && finite(out[i][k].x);
      for (let i = 1; i < out.length; i++) {
        if (okAt(i) || !okAt(i - 1)) continue;
        let j = i; while (j < out.length && !okAt(j)) j++;
        if (j < out.length && j - i <= maxRun) {
          const p = out[i - 1][k], q = out[j][k];
          for (let m = i; m < j; m++) {
            const w = (m - i + 1) / (j - i + 1);
            if (!out[m]) out[m] = [];
            out[m][k] = { x: p.x + (q.x - p.x) * w, y: p.y + (q.y - p.y) * w, visibility: .3 };
          }
        }
        i = j;
      }
    }
    return { frames: out, info };
  }
  // Name the side-view legs from the front/rear video, where MediaPipe's
  // left/right is reliable: at any moment the stance foot is the lower one in
  // both views.  Returns true when the side-view legs had to be swapped, false
  // when they already match, null when it cannot be told.
  // Lower foot (larger image y): stance foot.  Side view: heel/toe; front
  // or rear view: ankle/heel/toe.  Positive = the left foot is the lower one.
  function lowFootDiff(f, side) {
    const low = ks => { const ys = ks.map(k => f && f[k] && ok(f[k]) ? f[k].y : NaN).filter(finite); return ys.length ? Math.max(...ys) : NaN; };
    return side ? low([LANDMARK.LEFT_HEEL, LANDMARK.LEFT_TOE]) - low([LANDMARK.RIGHT_HEEL, LANDMARK.RIGHT_TOE])
      : low([LANDMARK.LEFT_ANKLE, LANDMARK.LEFT_HEEL, LANDMARK.LEFT_TOE]) - low([LANDMARK.RIGHT_ANKLE, LANDMARK.RIGHT_HEEL, LANDMARK.RIGHT_TOE]);
  }
  // Correct the tracked side-view legs step by step against the front/rear
  // video.  Between two crossings one leg stays in front; if a crossing was
  // missed or invented, every later step has the legs the wrong way round.
  // For each step (crossing to crossing), sum over its frames how well "which
  // foot is lower" agrees between the views, smooth over 5 steps, and swap the
  // steps where it disagrees.  Returns the number of steps swapped.
  // Swap a step only when the views clearly disagree around it (mean
  // agreement below -0.3 on a -1..1 scale); chosen on the Twente trials
  // (0.2-0.3 best for Heavy, 0.3 also safe for Full).
  const FRONTAL_FIX_MIN = 0.3;
  function fixLegsByFrontal(sideFrames, sideFps, frontal, crossingAt) {
    if (!crossingAt || crossingAt.length < 4) return 0;
    const bounds = [0].concat(crossingAt, [sideFrames.length]), score = [], count = [];
    for (let k = 0; k + 1 < bounds.length; k++) {
      let sum = 0, n = 0;
      for (let i = bounds[k]; i < bounds[k + 1]; i++) {
        const ds = lowFootDiff(sideFrames[i], true), df = lowFootDiff(frontal.frames[Math.floor((i + .5) / sideFps * frontal.fps)], false);
        if (finite(ds) && finite(df)) { sum += Math.sign(ds) * Math.sign(df); n++; }
      }
      score.push(sum); count.push(n);
    }
    let swapped = 0;
    for (let k = 0; k < score.length; k++) {
      const w = [Math.max(0, k - 2), k + 3], n = count.slice(...w).reduce((a, b) => a + b, 0);
      const agree = n ? score.slice(...w).reduce((a, b) => a + b, 0) / n : 0;   // -1 .. 1
      if (agree < -FRONTAL_FIX_MIN) { swapLegs(sideFrames.slice(bounds[k], bounds[k + 1])); swapped++; }
    }
    return swapped;
  }
  function nameLegsByFrontal(sideFrames, sideFps, frontal) {
    const low = (f, ks) => { const ys = ks.map(k => f && f[k] && ok(f[k]) ? f[k].y : NaN).filter(finite); return ys.length ? Math.max(...ys) : NaN; };
    const a = [], b = [];
    sideFrames.forEach((f, i) => {
      const g = frontal.frames[Math.floor((i + .5) / sideFps * frontal.fps)];
      const ds = low(f, [LANDMARK.LEFT_HEEL, LANDMARK.LEFT_TOE]) - low(f, [LANDMARK.RIGHT_HEEL, LANDMARK.RIGHT_TOE]);
      const df = low(g, [LANDMARK.LEFT_ANKLE, LANDMARK.LEFT_HEEL, LANDMARK.LEFT_TOE]) - low(g, [LANDMARK.RIGHT_ANKLE, LANDMARK.RIGHT_HEEL, LANDMARK.RIGHT_TOE]);
      if (finite(ds) && finite(df)) { a.push(ds); b.push(df); }
    });
    if (a.length < sideFps * 2) return null;
    const ma = a.reduce((s, v) => s + v, 0) / a.length, mb = b.reduce((s, v) => s + v, 0) / b.length;
    let n = 0, da = 0, db = 0;
    a.forEach((v, i) => { n += (v - ma) * (b[i] - mb); da += (v - ma) ** 2; db += (b[i] - mb) ** 2; });
    const r = da && db ? n / Math.sqrt(da * db) : 0;
    return Math.abs(r) < .2 ? null : { swap: r < 0, r };
  }
  // Side-view legs: the tracked version (cleanLegs) or the pose model's own
  // labels, whichever matches the front/rear video better, named from it.
  // MediaPipe labels some runners correctly and others by position; the
  // tracking fixes the second kind but can spoil the first.  Without a
  // front/rear video the tracked version is used (better on average).
  function resolveSideLegs(rawFrames, fps, frontal) {
    const cleaned = cleanLegs(rawFrames, fps);
    const out = { frames: cleaned.frames, method: "tracked", naming: null, info: cleaned.info };
    if (!frontal) return out;
    out.info.stepsSwapped = fixLegsByFrontal(cleaned.frames, fps, frontal, cleaned.info.crossingAt);
    const raw = rawFrames.map(f => f ? f.slice() : null);
    const nt = nameLegsByFrontal(cleaned.frames, fps, frontal), nr = nameLegsByFrontal(raw, fps, frontal);
    const strength = n => n ? Math.abs(n.r) : 0;
    if (strength(nr) > strength(nt)) { out.frames = raw; out.method = "labels"; out.naming = nr; }
    else out.naming = nt;
    if (out.naming && out.naming.swap) swapLegs(out.frames);
    return out;
  }
  function swapLegs(frames) {
    frames.forEach(f => { if (f) LEG_CHAIN.L.forEach((k, j) => { const r = LEG_CHAIN.R[j]; [f[k], f[r]] = [f[r], f[k]]; }); });
  }
  function percentile(values, q) {
    const a = values.filter(finite).sort((x, y) => x - y);
    return a.length ? a[Math.min(a.length - 1, Math.max(0, Math.round(q * (a.length - 1))))] : NaN;
  }
  // Heel's forward peaks, measured from the pelvis so that the runner
  // drifting forward or back on the treadmill does not move the threshold.
  function contactsFor(frames, leg, fps, direction) {
    const heel = leg === "L" ? LANDMARK.LEFT_HEEL : LANDMARK.RIGHT_HEEL;
    const raw = frames.map(f => {
      const p = point(f, heel), l = point(f, LANDMARK.LEFT_HIP), r = point(f, LANDMARK.RIGHT_HIP);
      return ok(p) && ok(l) && ok(r) ? direction * (p.x - (l.x + r.x) / 2) : NaN;
    });
    const smoothed = movingAverage(raw, Math.max(1, Math.round(fps * 0.04)));
    const lo = percentile(smoothed, .05), hi = percentile(smoothed, .95);
    if (!finite(lo) || !finite(hi)) return [];
    return localMaxima(smoothed, Math.max(3, Math.round(fps * 0.45)), lo + (hi - lo) * 0.50);
  }
  // Frame (fractional) at which the toe is furthest behind the pelvis,
  // searched in [from, to).  Parabolic interpolation around the minimum.
  function toeBackFrame(frames, leg, from, to, fps, direction) {
    const toe = leg === "L" ? LANDMARK.LEFT_TOE : LANDMARK.RIGHT_TOE;
    const rel = movingAverage(frames.slice(from, to).map(f => {
      const p = point(f, toe), l = point(f, LANDMARK.LEFT_HIP), r = point(f, LANDMARK.RIGHT_HIP);
      return ok(p) && ok(l) && ok(r) ? direction * (p.x - (l.x + r.x) / 2) : NaN;
    }), Math.max(1, Math.round(fps * .03)));
    let b = -1;
    rel.forEach((v, i) => { if (finite(v) && (b < 0 || v < rel[b])) b = i; });
    if (b <= 0 || b >= rel.length - 1) return null;  // at the window edge: not a real minimum
    const curve = rel[b - 1] - 2 * rel[b] + rel[b + 1];
    return from + b + (finite(curve) && curve > 0 ? .5 * (rel[b - 1] - rel[b + 1]) / curve : 0);
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
  // Frontal-plane angles, defined as seen from behind (spec section 3).  A
  // front view mirrors the image, which flips only KA's lateral direction:
  // CPD uses the vertical offset and |dx|, and HADD is an unsigned angle.
  // Same convention as rtmpose_pipeline --rear-view front.
  function rearValues(frame, leg, view) {
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
    const outside = (leg === "L" ? -1 : 1) * (knee.x - lineX) * (view === "front" ? -1 : 1);
    return { CPD: cpd, HADD: hadd, KA: Math.sign(outside || 1) * magnitude };
  }
  function emptyLeg() {
    return { CPD: null, HADD: null, KA: null, KF: null, knee_flex_contact: null,
      shank_angle_contact: null, foot_angle_contact: null, overstride: null, n_steps: 0 };
  }
  /* ---- MP4 / QuickTime header reader (frame rate, frame count, codec) ----
   * Browsers do not expose a video's frame rate, so read it from the first
   * video track: mdhd gives the timescale and stts the per-frame durations.
   * Only the box headers and the moov box are read, never the media data. */
  async function readBytes(file, start, end) { return new Uint8Array(await file.slice(start, end).arrayBuffer()); }
  function fourcc(u8, at) { return String.fromCharCode(u8[at], u8[at + 1], u8[at + 2], u8[at + 3]); }
  function childBoxes(u8, start, end) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength), out = [];
    for (let o = start; o + 8 <= end;) {
      let size = dv.getUint32(o), header = 8;
      if (size === 1) { size = dv.getUint32(o + 8) * 4294967296 + dv.getUint32(o + 12); header = 16; }
      else if (size === 0) size = end - o;
      if (size < header || o + size > end) break;
      out.push({ type: fourcc(u8, o + 4), start: o + header, end: o + size });
      o += size;
    }
    return out;
  }
  function child(u8, box, type) { return box && childBoxes(u8, box.start, box.end).find(b => b.type === type); }
  async function findMoov(file) {
    for (let o = 0, guard = 0; o + 8 <= file.size && guard < 10000; guard++) {
      const h = await readBytes(file, o, Math.min(file.size, o + 16)), dv = new DataView(h.buffer);
      let size = dv.getUint32(0), header = 8;
      if (size === 1 && h.length >= 16) { size = dv.getUint32(8) * 4294967296 + dv.getUint32(12); header = 16; }
      else if (size === 0) size = file.size - o;
      if (size < header) return null;
      if (fourcc(h, 4) === "moov") return size > 64 * 1024 * 1024 ? null : readBytes(file, o + header, o + size);
      o += size;
    }
    return null;
  }
  async function probeVideo(file) {
    try {
      const moov = await findMoov(file);
      if (!moov) return null;
      const dv = new DataView(moov.buffer, moov.byteOffset, moov.byteLength);
      for (const trak of childBoxes(moov, 0, moov.length).filter(b => b.type === "trak")) {
        const mdia = child(moov, trak, "mdia"), hdlr = child(moov, mdia, "hdlr");
        if (!hdlr || fourcc(moov, hdlr.start + 8) !== "vide") continue;
        const mdhd = child(moov, mdia, "mdhd"), stbl = child(moov, child(moov, mdia, "minf"), "stbl");
        const stts = child(moov, stbl, "stts"), stsd = child(moov, stbl, "stsd");
        if (!mdhd || !stts) return null;
        const timescale = dv.getUint32(mdhd.start + (moov[mdhd.start] === 1 ? 20 : 12));
        const entries = [];
        for (let i = 0, n = dv.getUint32(stts.start + 4); i < n && stts.start + 16 + 8 * i <= stts.end; i++) {
          entries.push({ count: dv.getUint32(stts.start + 8 + 8 * i), delta: dv.getUint32(stts.start + 12 + 8 * i) });
        }
        const frames = entries.reduce((s, e) => s + e.count, 0), ticks = entries.reduce((s, e) => s + e.count * e.delta, 0);
        if (!timescale || !frames || !ticks) return null;
        // Nominal rate from the most common frame duration; a short final
        // entry is normal and does not make the clip variable-rate.
        const byDelta = new Map();
        entries.forEach(e => byDelta.set(e.delta, (byDelta.get(e.delta) || 0) + e.count));
        const mode = [...byDelta.entries()].sort((a, b) => b[1] - a[1])[0][0];
        const offRate = entries.slice(0, -1).filter(e => Math.abs(e.delta - mode) > mode * .01).reduce((s, e) => s + e.count, 0);
        const variable = offRate > frames * .05;
        return { fps: variable ? frames * timescale / ticks : timescale / mode, frames, duration: ticks / timescale,
          variable, codec: stsd && stsd.start + 16 <= stsd.end ? fourcc(moov, stsd.start + 12) : null };
      }
      return null;
    } catch (_) { return null; }
  }
  async function importVision() {
    // This is the package's browser module entry point.  Do not use
    // jsDelivr's /+esm conversion endpoint here: it does not reliably expose
    // MediaPipe's WASM-aware package entry on all browsers.
    return import(config.assetBaseUrl + "/vision_bundle.mjs");
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
    const okHandler = () => { cleanup(); resolve(); }, fail = () => {
      cleanup();
      const code = target.error && target.error.code;
      const detail = ({ 1: "the file was aborted", 2: "a network error occurred", 3: "the video codec is not supported or the file is corrupt", 4: "the browser does not support this video format" })[code] || "the browser could not decode it";
      reject(new Error("Could not read the selected video: " + detail + ". Use an H.264/AVC MP4 (avc1) with AAC audio."));
    };
    const cleanup = () => { target.removeEventListener(type, okHandler); target.removeEventListener("error", fail); };
    target.addEventListener(type, okHandler, { once: true }); target.addEventListener("error", fail, { once: true });
  }); }
  async function rejectUnsupportedMp4v(file) {
    // The supplied reference clips are MPEG-4 Part 2 (mp4v).  Chrome does not
    // generally decode that codec, even though its file extension is .mp4.
    const size = Math.min(file.size, 256 * 1024);
    const chunks = await Promise.all([file.slice(0, size).arrayBuffer(), file.slice(Math.max(0, file.size - size)).arrayBuffer()]);
    const text = new TextDecoder("latin1").decode(new Uint8Array(chunks[0])) + new TextDecoder("latin1").decode(new Uint8Array(chunks[1]));
    if (text.includes("mp4v")) throw new Error("This MP4 uses MPEG-4 Part 2 (mp4v), which Chrome cannot analyse. Re-encode it as H.264/AVC (avc1) MP4, then select the converted file.");
  }
  async function openVideo(file, codec) {
    if (codec === "mp4v") throw new Error("This MP4 uses MPEG-4 Part 2 (mp4v), which Chrome cannot analyse. Re-encode it as H.264/AVC (avc1) MP4, then select the converted file.");
    if (!codec) await rejectUnsupportedMp4v(file);
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
  async function inferVideo(file, timing, maxSeconds, progress, progressBase, progressSpan, signal, timestampOffsetMs) {
    const { video, url } = await openVideo(file, timing.codec);
    try {
      const landmarker = await getLandmarker();
      const fps = timing.fps, duration = Math.min(video.duration, maxSeconds);
      const available = timing.frames ? Math.min(timing.frames, Math.floor(video.duration * fps + 1e-6)) : Math.floor(video.duration * fps + 1e-6);
      const count = Math.max(2, Math.min(available, Math.floor(duration * fps + 1e-6)));
      // frames: image points (0-1).  world: MediaPipe's 3-D estimate in metres,
      // origin between the hips; used for the leg the side camera cannot see.
      const frames = [], world = [];
      let lastTs = -Infinity;
      for (let i = 0; i < count; i++) {
        if (signal && signal.aborted) throw new DOMException("Analysis cancelled", "AbortError");
        // Seek to the middle of frame i, not its start: a seek exactly on a
        // frame boundary can return the previous frame after rounding.
        await seek(video, Math.min(video.duration - .001, (i + .5) / fps));
        // MediaPipe requires timestamps to increase for the lifetime of one
        // VIDEO-mode landmarker.  Rear and side files share that instance.
        const ts = Math.max(lastTs + 1, Math.round((timestampOffsetMs || 0) + i * 1000 / fps));
        lastTs = ts;
        const result = landmarker.detectForVideo(video, ts);
        frames.push(result.landmarks && result.landmarks[0] ? result.landmarks[0] : null);
        world.push(result.worldLandmarks && result.worldLandmarks[0] ? result.worldLandmarks[0] : null);
        if (i % 3 === 0 || i === count - 1) progress(progressBase + progressSpan * (i + 1) / count, "Estimating pose");
      }
      return { frames, world, fps, duration: count / fps, lastTs, originalDuration: video.duration };
    } finally { URL.revokeObjectURL(url); }
  }
  // Steps as fractional frame indices: initial contact, toe-off, the
  // following contralateral contact, and the next contact of the same leg.
  // Cycles are found from the heel's forward extremes; the two event times
  // are then shifted by the corrections above.
  function detectSteps(sideFrames, fps, direction) {
    const peaks = { L: contactsFor(sideFrames, "L", fps, direction), R: contactsFor(sideFrames, "R", fps, direction) };
    const allPeaks = Object.entries(peaks).flatMap(([leg, xs]) => xs.map(frame => ({ frame, leg }))).sort((a, b) => a.frame - b.frame);
    const shift = IC_AFTER_HEEL_FORWARD_S * fps, steps = [];
    for (const leg of ["L", "R"]) for (let i = 0; i + 1 < peaks[leg].length; i++) {
      const peak = peaks[leg][i], nextPeak = peaks[leg][i + 1];
      const opposite = allPeaks.find(e => e.frame > peak && e.leg !== leg);
      if (!opposite || opposite.frame >= nextPeak) continue;
      // The toe is furthest back shortly before or after the opposite foot
      // lands (about 0.17 stride after toe-off), so search only up to 30% of
      // the stride past that landing.
      const back = toeBackFrame(sideFrames, leg, peak, Math.min(nextPeak, Math.round(opposite.frame + .3 * (nextPeak - peak)) + 1), fps, direction);
      if (back == null) continue;
      const contact = peak + shift, next = nextPeak + shift;
      const off = back - TO_BEFORE_TOE_BACK_STRIDE * (nextPeak - peak);
      if (!(off > contact && off < next)) continue;
      steps.push({ leg, contact, off, opposite: opposite.frame + shift, next });
    }
    return steps;
  }
  function summarize(sideFrames, rear, fps, timingKnown, heightCm, notes) {
    const rearProvided = Boolean(rear);
    const direction = progressionSign(sideFrames);
    const values = { L: [], R: [] }, steps = { L: [], R: [] }, duties = { L: [], R: [] }, pelvisOsc = [];
    const headToFoot = sideFrames.map(f => {
      const nose = point(f, LANDMARK.NOSE), lh = point(f, LANDMARK.LEFT_HEEL), rh = point(f, LANDMARK.RIGHT_HEEL);
      return ok(nose) && (ok(lh) || ok(rh)) ? Math.max((lh || rh).y, (rh || lh).y) - nose.y : NaN;
    });
    const heightPx = median(headToFoot);
    const mmPerUnit = finite(heightCm) && finite(heightPx) && heightPx > 0 ? heightCm * 10 / heightPx : NaN;
    for (const { leg, contact, off, opposite: oppositeFrame } of detectSteps(sideFrames, fps, direction)) {
      const opposite = { frame: oppositeFrame };
      const at = f => Math.min(sideFrames.length - 1, Math.round(f)), mid = at((contact + off) / 2), ic = at(contact), stepS = (opposite.frame - contact) / fps, contactS = (off - contact) / fps;
      const sideMid = sideValues(sideFrames[mid], leg, direction), sideContact = sideValues(sideFrames[ic], leg, direction);
      const rearRow = rearProvided ? rearValues(frameAt(rear, mid, fps), leg, rear.view) : { CPD: NaN, HADD: NaN, KA: NaN };
      const interval = sideFrames.slice(ic, at(opposite.frame) + 1).map(f => {
        const l = point(f, LANDMARK.LEFT_HIP), r = point(f, LANDMARK.RIGHT_HIP); return ok(l) && ok(r) ? (l.y + r.y) / 2 : NaN;
      }).filter(finite);
      if (interval.length && finite(mmPerUnit)) pelvisOsc.push((Math.max(...interval) - Math.min(...interval)) * mmPerUnit);
      const row = { ...rearRow, KF: sideMid.KF, knee_flex_contact: sideContact.knee_flex_contact,
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
      // KF is not reported: against motion capture (Twente, 24 trials) the
      // 2-D side view could not rank runners by mid-stance knee flexion
      // (r <= 0.4, about 15-19 degrees too small; validation/README.md).
      legs[leg].KF = null;
    }
    const all = values.L.concat(values.R);
    const strike = leg => { const a = median(values[leg].map(v => v.foot_angle_contact)); return finite(a) ? (a > 5 ? "rearfoot" : "non_rearfoot") : null; };
    const lStrike = strike("L"), rStrike = strike("R");
    function asym(a, b) { const x = median(a), y = median(b); return finite(x) && finite(y) && x + y ? Math.abs(x - y) / ((x + y) / 2) * 100 : null; }
    notes.push("Knee flexion at mid-stance (KF) is not reported: it could not be measured reliably from video in validation; the load estimate uses the study median instead.");
    if (!rearProvided) notes.push("No rear or front video was supplied; CPD, HADD, and KA are unavailable.");
    if (Math.min(values.L.length, values.R.length) < MIN_STEPS) notes.push("Fewer than seven valid steps per leg were measured; record a longer, clearer trial.");
    if (!timingKnown) {
      // Without the capture rate the time resolution is unknown, so the short
      // intervals (about 0.1-0.3 s) are withheld.  Cadence spans a whole step
      // and is kept.
      all.forEach(v => { v.contact_s = NaN; v.flight_s = NaN; v.duty = NaN; });
      Object.values(duties).forEach(a => a.fill(NaN));
      notes.push("The capture frame rate could not be read from the file and was not selected, so contact time, flight time, and duty factor are not reported. Select the recording fps and analyse again.");
    }
    if (!finite(heightPx) || heightPx < .5) notes.push("Full-body height could not be estimated reliably; pelvis vertical oscillation may be unavailable.");
    return {
      rhythm: { cadence_spm: round((() => { const v = median(all.map(s => s.step_s)); return finite(v) && v > 0 ? 60 / v : NaN; })()),
        contact_s: round(median(all.map(s => s.contact_s))), flight_s: round(median(all.map(s => s.flight_s))), duty: round(median(all.map(s => s.duty))),
        foot_strike: lStrike === rStrike ? lStrike : null, cadence_asym_pct: round(asym(steps.L, steps.R)), duty_asym_pct: round(asym(duties.L, duties.R)),
        alt_strike: lStrike && rStrike ? lStrike !== rStrike : null, pelvis_vertical_osc_mm: round(median(pelvisOsc)) },
      legs, quality: { step_sd: Object.fromEntries(metricNames.concat(["contact_s", "flight_s", "duty"]).map(k => [k, k === "KF" ? null : round(sd(all.map(v => v[k])))])),
        low_confidence_frames_pct: null, notes }
    };
  }
  // Capture rate for one file: the file header first, the user's choice
  // only when the header cannot be read.
  async function resolveTiming(file, userFps, label, notes) {
    const meta = await probeVideo(file);
    if (meta && meta.fps >= 10 && meta.fps <= 1000) {
      if (meta.variable) notes.push(`The ${label} video has a variable frame rate; timing uses its average of ${round(meta.fps, 1)} fps.`);
      if (finite(userFps) && Math.abs(userFps - meta.fps) > 1) notes.push(`The ${label} video header says ${round(meta.fps, 2)} fps; that was used instead of the selected ${userFps} fps.`);
      return { fps: meta.fps, frames: meta.frames, codec: meta.codec, source: "metadata" };
    }
    if (finite(userFps)) return { fps: userFps, frames: null, codec: meta && meta.codec, source: "user" };
    return { fps: FALLBACK_SAMPLING_FPS, frames: null, codec: meta && meta.codec, source: null };
  }
  async function extractMeasures(files, subject, options) {
    const rearFile = files && files.rear, sideFile = files && files.side;
    if (!sideFile) throw new Error("A side-view video is required to measure timing and running form.");
    options = options || {}; const progress = typeof options.onProgress === "function" ? options.onProgress : () => {};
    const frontalView = options.frontalView === "front" ? "front" : "rear";
    const userFps = finite(options.fps) && options.fps >= 10 ? options.fps : null;
    const maxSeconds = finite(options.maxSeconds) && options.maxSeconds > 0 ? options.maxSeconds : MAX_SECONDS;
    const notes = [];
    const sideTiming = await resolveTiming(sideFile, userFps, "side", notes);
    const rearTiming = rearFile ? await resolveTiming(rearFile, userFps, frontalView, notes) : null;
    const timingKnown = sideTiming.source !== null;
    if (timingKnown && sideTiming.fps < 60) {
      notes.push(`The side video is ${round(sideTiming.fps, 1)} fps, so contact and flight times are only resolved to one frame (${Math.round(1000 / sideTiming.fps)} ms). Record at 120 fps or more for timing.`);
    }
    progress(.01, "Loading pose model"); await getLandmarker();
    const side = await inferVideo(sideFile, sideTiming, maxSeconds, progress, .03, rearFile ? .48 : .94, options.signal, 0);
    const rear = rearFile ? await inferVideo(rearFile, rearTiming, maxSeconds, progress, .51, .43, options.signal, side.lastTs + 1000) : null;
    if (side.originalDuration > maxSeconds || (rear && rear.originalDuration > maxSeconds)) {
      notes.push(`Only the first ${maxSeconds} seconds of each video were analysed; trim the recording to a steady running section.`);
    }
    if (rear) {
      rear.view = frontalView;
      if (frontalView === "front") notes.push("The frontal-plane video was recorded from the front; KA was mirrored to the rear-view sign convention. Front-view CPD, HADD, and KA have not yet been validated against rear-view measurements.");
      notes.push("Frontal and side videos were aligned from their first frames. Use recordings that start at the same moment for valid frontal-plane angles.");
      if (Math.abs(rear.originalDuration - side.originalDuration) > 1 / sideTiming.fps + .02) {
        notes.push(`The ${frontalView} and side videos differ in length (${round(rear.originalDuration, 2)} s vs ${round(side.originalDuration, 2)} s), so they may not start together.`);
      }
    }
    progress(.96, "Calculating running measures");
    const rawSide = side.frames, legs = resolveSideLegs(rawSide, sideTiming.fps, rear);
    side.frames = legs.frames;
    if (legs.method === "tracked" && legs.info.crossingFrames) notes.push(`Side view: left and right legs were tracked through ${legs.info.crossingFrames} leg-crossing frames by motion continuity; ${legs.info.predicted} points pulled onto the other leg were replaced by the predicted path.`);
    const naming = legs.naming;
    if (!naming) notes.push(rear ? "Side view: the left and right legs could not be matched to the " + frontalView + " video, so side-view left and right values may be swapped." : "Side view: without a front or rear video, which leg is left cannot be confirmed; side-view left and right values may be swapped.");
    const measured = summarize(side.frames, rear, sideTiming.fps, timingKnown, numeric(subject && subject.height_cm), notes);
    const steps = detectSteps(side.frames, sideTiming.fps, progressionSign(side.frames));
    progress(1, "Done");
    // One source for both files: the least certain of the two.
    const sources = [sideTiming.source].concat(rearTiming ? [rearTiming.source] : []);
    const fpsSource = sources.includes(null) ? null : sources.includes("user") ? "user" : "metadata";
    // `rear` keeps its name from the measures.json contract; `frontal_view`
    // says which side of the runner that file was recorded from.
    const result = { video: { rear: rearFile ? rearFile.name : null, side: sideFile.name, frontal_view: rearFile ? frontalView : null,
        fps_rear: rearTiming && rearTiming.source ? round(rearTiming.fps, 3) : null,
        fps_side: timingKnown ? round(sideTiming.fps, 3) : null, fps_source: fpsSource },
      subject: { speed_kmh: numeric(subject && subject.speed_kmh), height_cm: numeric(subject && subject.height_cm),
        mass_kg: numeric(subject && subject.mass_kg), sex: subject && subject.sex || null },
      ...measured, model: { pose: "MediaPipe Pose Landmarker heavy (" + MP_VERSION + ")", extractor: VERSION } };
    // Per-frame points and detected events for drawing over the videos.  Not
    // enumerable, so JSON.stringify(result) is still a plain measures.json.
    // Times are seconds on each file's own timeline; frame i covers
    // [i / fps, (i + 1) / fps).
    const pts = frames => frames.map(f => f ? Array.from(f, p => p ? [p.x, p.y, p.visibility == null ? 1 : p.visibility, finite(p.z) ? p.z : null] : null) : null);
    const sec = f => f / sideTiming.fps;
    Object.defineProperty(result, "overlay", { enumerable: false, value: {
      side: { fps: sideTiming.fps, frames: pts(side.frames), raw: pts(rawSide), world: pts(side.world) },
      frontal: rear ? { fps: rearTiming.fps, frames: pts(rear.frames), world: pts(rear.world), view: frontalView } : null,
      steps: steps.map(s => ({ leg: s.leg, contact: sec(s.contact), off: sec(s.off), opposite: sec(s.opposite) })),
    } });
    return result;
  }
  root.StrideExtract = { available: true, version: VERSION,
    configure(next) { config = Object.assign({}, config, next || {}); landmarkerPromise = null; }, extractMeasures, probeVideo,
    _internal: { TO_BEFORE_TOE_BACK_STRIDE, IC_AFTER_HEEL_FORWARD_S, rearValues, detectSteps, progressionSign, cleanLegs, nameLegsByFrontal, resolveSideLegs, summarize, sideValues, swapLegs } };
})(typeof self !== "undefined" ? self : this);
