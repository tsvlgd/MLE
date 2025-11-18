# ml/ — Learning, Notes, and Implementations

Purpose: `ml/` is the canonical place for learning material, tutorials, implementations, and lightweight projects. It should contain both polished docs (for readers) and raw notes (for personal reference) in separate subfolders.

Quick index (create or verify these exist):

- `ml/01-docs/docs/` — Polished tutorials, course-like writeups intended for readers.
- `ml/03-notes/notes/` — Raw personal notes, book notes, exploratory writeups.
- `ml/02-implementations/implementations/` — Code and notebooks with algorithm implementations.
- `ml/04-kaggle/kaggle/` — Competition exercises.
- `ml/05-projects/projects/` — End-to-end projects and pipelines.
- `ml/06-scripts/scripts/` and `ml/07-utils/utils/` — Small reusable code modules.

How I suggest you use this folder:
- When experimenting, put drafts in `ml/03-notes/notes/`.
- When something becomes a tutorial or stable how-to, move the notebook to `ml/01-docs/docs/` and update the `ml/01-docs/docs/README.md` with a short description.
- Keep datasets in `datasets/` and reference them from notebooks using relative paths documented in `ml/README.md`.

If you accept this organization I can prepare a small branch with recommended `git mv` commands for a subset of files you point out.
