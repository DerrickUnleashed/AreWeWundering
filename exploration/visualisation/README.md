# EDA — `exploration/visualisation/`

EDA scripts live here; figures go in `images/`.

**Rules, so this folder does not become a graveyard of curiosity plots:**

1. **Every script starts with a docstring naming the decision it informs**, in the
   form *"Decision: <the question>. If the answer is A we do X, if B we do Y."*
   A script that cannot name its decision does not get written.
2. **Output naming: `<topic>_<what>.png`** — e.g. `weight_concentration_by_decile.png`,
   `scored_row_positions.png`, `target_autocorrelation.png`. Lowercase, underscores,
   no spaces, no dates in the name (git history is the date).
3. **No plots that do not change a decision.** The Phase 1 question list is in
   `AGENTS.md` §12; the only reason to plot something is to answer one of those
   questions or a question raised later by a result.
4. **Train-only statistics by default.** Anything computed from `valid.parquet`
   must say so in the docstring, and must not feed a model input. The
   normalisation statistics live in `cache/norm_stats.json` (train split only).
5. **Read the shards, not the parquet**, once they exist — memory-map them:
   ```python
   X = np.load('cache/train/X.npy', mmap_mode='r')          # (212140000, 112) float16
   y = np.load('cache/train/y.npy', mmap_mode='r')          # (212140000, 2)   float32
   seq_index = np.load('cache/train/seq_index.npy')         # 10608 start offsets
   ```
   `cache/` holds the full 10,607 train + 1,873 validation sequences as float16
   memmaps (see `tools/make_shards.py`), so no EDA script should ever open
   `datasets/*.parquet` unless it is specifically about the parquet layer.
6. **A finding gets written into `AGENTS.md`** (usually §14) with the number and
   how it was obtained, or it does not exist. Next sessions only read that file.

Current status: no EDA scripts yet. EDA runs on request, driven by the Phase 1
questions in `AGENTS.md` §12, and is not created speculatively.
