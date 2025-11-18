# advanced_ml — Active learning track

Purpose: `advanced_ml/` is where you follow courses, active readings, or deeper explorations that are still "in-progress". Treat this folder as a "workbench". When a notebook or note becomes stable and polished, move it into `ml/01-docs/docs/` or `ml/02-implementations/implementations/`.

Conventions:
- Use clear folder names for each topic: `advanced_ml/01-backprop-by-hand/`, `advanced_ml/02-bert/`, etc.
- Each topic folder should have a `README.md` summarizing the goal and key files.
- Avoid duplicating finished tutorials in `ml/01-docs/docs/` and `advanced_ml/` — move the canonical copy into `ml/01-docs/docs/` and leave a short pointer in `advanced_ml/`.

Examples of housekeeping steps:
- If a topic is finished, run `git mv advanced_ml/07-hallucination/ advanced_ml/archive/07-hallucination/` or move it into `ml/01-docs/docs/`.
- Add `advanced_ml/status.md` or `README.md` per topic to show progress (TODO, In progress, Done).

If you'd like, I can scan `advanced_ml/` and propose a short list of candidate folders to promote into `ml/01-docs/docs/` (polished) or archive inside `advanced_ml/archive/`.
