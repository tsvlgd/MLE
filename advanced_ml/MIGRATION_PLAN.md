# advanced_ml Migration Plan (proposed)

Goal: reorganize `advanced_ml/` to follow a clear numbered-topic layout like the JINO-ROHIT example. Keep history (use git mv) and perform moves on a review branch.

Current relevant items (detected):
- `advanced_ml/01-neural-nets/neural_net/` — contains `notebooks/NN_from_scratch.ipynb` and `Readme.md`
- `advanced_ml/02-pytorch-opencv/pytorch-opencv/` — exists as a topic folder

Proposed target layout (example):
- `advanced_ml/01-backprop-by-hand/`  <- move `neural_net/` here (or create `01-neural-net` if you prefer)
- `advanced_ml/02-pytorch-opencv/`   <- move `pytorch-opencv/` here (rename to add numeric prefix)
- `advanced_ml/99-archive/`          <- any completed/past topics to archive

Suggested safe process (manual review required):
1. Create a branch:
   git checkout -b reorg/advanced_ml-numbered-topics

2. Create destination folders and move files with git mv (example):
   mkdir -p advanced_ml/01-backprop-by-hand
   git mv advanced_ml/01-neural-nets/neural_net advanced_ml/01-backprop-by-hand/neural_net

   mkdir -p advanced_ml/02-pytorch-opencv
   git mv advanced_ml/02-pytorch-opencv/pytorch-opencv advanced_ml/02-pytorch-opencv

3. Flatten inner structure where appropriate and add topic README files with purpose and links.
   e.g. move `advanced_ml/01-backprop-by-hand/neural_net/notebooks/NN_from_scratch.ipynb` up to `advanced_ml/01-backprop-by-hand/` if desired:
   git mv "advanced_ml/01-backprop-by-hand/neural_net/notebooks/NN_from_scratch.ipynb" "advanced_ml/01-backprop-by-hand/NN_from_scratch.ipynb"

4. Run the link-fixer script to update references within notebooks and README files (I can add this script).

5. Commit and push the branch, review changes and run tests (if any), then merge.

Notes & caveats:
- Use `git mv` to preserve history.
- After moving, search for broken references and update them: grep -R "neural_net/" -n
- I can prepare the exact `git mv` commands for each file once you confirm the final mapping.

If you give me a thumbs-up for this proposed mapping (or adjust names), I can:
- Create the numbered topic folders skeleton and topic README files (no file moves yet), and
- Add the `tools/fix_notebook_links.py` script and `tools/scaffold_topics.py` helper to automate scaffolding and link fixing on a branch.

Pick one: (A) scaffold-only now, or (B) scaffold and run the first safe move (e.g., move `neural_net` into `01-backprop-by-hand`) on a new branch. I'll proceed once you confirm.
