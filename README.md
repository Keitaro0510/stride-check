# Stride Check

Prototype from BioHackathon 2026. It shows running-style types, a form-based injury prediction and estimated load on four body parts from running measurements, and sketches how runners could donate data to grow an open prospective dataset.

**Live page:** https://keitaro0510.github.io/stride-check/

- Everything runs in the browser. Nothing you load is sent anywhere.
- Video analysis (`extract/`) is in development. For now, the page shows three sample runners and can load a `measures.json` file.
- The injury prediction is weak (AUC 0.64, 95% CI 0.52–0.76) and is not medical advice.

## Files
| Path | Contents |
|---|---|
| `index.html` | The page (English first, Japanese toggle) |
| `model.js` | All calculations: `StrideModel.evaluate(measures, APP_DATA)` |
| `i18n.js` | Text in English and Japanese |
| `data/app_data.js` | Type cut-offs, prediction and load model coefficients, reference distributions, samples |
| `data/samples/` | Sample `measures.json` files |
| `extract/` | Video → `measures.json` (placeholder; see `extract/README.md`) |
| `docs/video_measurement_spec.md` | Measurement definitions and requirements (Japanese) |

## Data sources
- Wu et al. 2026, *npj Digital Medicine*: 142 endurance runners, 12-month prospective follow-up (CC BY 4.0). Rhythm types and distributions.
- Loh et al. 2025, *Int J Sports Med*: 81 runners, 12-month prospective follow-up (CC BY-NC). Form types and injury prediction. Only summary statistics and model coefficients are included; no individual records. Non-commercial use only.
- Fukuchi et al. 2017, *PeerJ*: 3D motion and ground reaction forces of 39 runners (CC BY 4.0). Load models and the three sample runners.

The analysis code that produced these models is kept in a separate repository.
