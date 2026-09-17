# extract/ — video → measures.json

This folder is where the video analysis goes. The page (`index.html`) loads `extract/extract.js`, which must define `window.StrideExtract`.

```js
window.StrideExtract = {
  available: true,                 // false shows "in development" and disables the button
  version: "0.1.0",
  async extractMeasures({ rear, side }, subject, { onProgress }) {
    // rear, side: File objects chosen by the user (either may be null)
    // subject: { speed_kmh, height_cm, mass_kg, sex } typed on the page (null if empty)
    // onProgress(p, message): p from 0 to 1
    return measures;               // same format as measures.json
  },
};
```

- The returned object is passed to `StrideModel.evaluate(measures, APP_DATA)` in `model.js`. See `data/samples/sample_1.json` for a complete example.
- Definitions, sign conventions, accuracy targets and browser requirements (model formats, fps, codecs, processing time, licences) are in `video_measurement_spec.md`, sections 2, 3, 4 and 7.
- If the video cannot be analysed well (body not fully visible, too few steps), return what you can and explain in `quality.notes`. The page shows the notes.
- Put model files (ONNX, TensorFlow.js, MediaPipe `.task`) and helper scripts in this folder and load them with relative paths.
- Nothing may be sent off the device. Loading model files from this site is fine.
