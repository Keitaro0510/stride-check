/* Video → measures.json (placeholder).
 *
 * Replace this file with the real implementation. The page calls:
 *
 *   const measures = await window.StrideExtract.extractMeasures(
 *     { rear: File | null, side: File | null },
 *     { speed_kmh, height_cm, mass_kg, sex },          // values typed on the page (null if empty)
 *     { onProgress: (p, message) => {} }                // p = 0..1
 *   );
 *
 * and passes the result to StrideModel.evaluate(). The return value must follow the measures.json
 * format in video_measurement_spec.md (sections 2, 3 and 7). Unmeasured values are null.
 * Set `available: true` once extractMeasures works.
 */
(function (root) {
  "use strict";
  root.StrideExtract = {
    available: false,
    version: "placeholder",
    async extractMeasures() {
      throw new Error("video analysis is not implemented yet");
    },
  };
})(typeof self !== "undefined" ? self : this);
