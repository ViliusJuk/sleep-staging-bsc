# Sleep Staging (Sleep-EDF, Fpz-Cz, 30s) – BSc

Pipeline: duomenų paruošimas → treniravimas → testavimas → metrikos (Accuracy, Macro-F1, Cohen’s κ) → Confusion Matrix + grafikai.

## Aplankai
- `data/raw` – originalūs EDF/Hypnogram failai (Sleep-EDF Expanded)
- `data/processed` – su-epochinti ir išvalyti .npz
- `models` – išsaugoti svoriai `.pth`
- `results` – grafikai ir ataskaitos
- `src/sleepstaging` – kodas (paruošimas, dataset, modeliai)
- `run_sleep_staging.py` – testavimo/analizės paleidimo skriptas
