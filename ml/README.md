# ml/ — Learning, Notes, and Implementations

Purpose: `ml/` is the canonical place for learning material, tutorials, implementations, and lightweight projects. It should contain both polished docs (for readers) and raw notes (for personal reference) in separate subfolders.

Quick index (create or verify these exist):

- `ml/docs/` — Polished tutorials, course-like writeups intended for readers.
- `ml/notes/` — Raw personal notes, book notes, exploratory writeups.
- `ml/implementations/` — Code and notebooks with algorithm implementations.
- `ml/kaggle/` — Competition exercises.
- `ml/projects/` — End-to-end projects and pipelines.
- `ml/scripts/` and `ml/utils/` — Small reusable code modules.

How I suggest you use this folder:
- When experimenting, put drafts in `ml/notes/`.
- When something becomes a tutorial or stable how-to, move the notebook to `ml/docs/` and update the `ml/docs/README.md` with a short description.
- Keep datasets in `datasets/` and reference them from notebooks using relative paths documented in `ml/README.md`.

If you accept this organization I can prepare a small branch with recommended `git mv` commands for a subset of files you point out.
