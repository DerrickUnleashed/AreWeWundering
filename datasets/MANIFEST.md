# Dataset manifest

The two large parquet files are **not** in git, and cannot be. The small one is.

| File | Size | In git? | Reason |
|---|---:|---|---|
| `datasets/train.parquet` | 29.2 GB | no | over both limits (see below) |
| `datasets/valid.parquet` | 5.6 GB | no | over both limits (see below) |
| `datasets/valid_mask.parquet` | 92 MB | **yes**, as a normal git blob | under the 100 MB per-blob limit |

## Why the two large files are out

Both candidate mechanisms were tried, and both fail:

- **Regular git**: GitHub rejects any file over **100 MB**.
- **git-lfs** (configured via `.gitattributes` on 2026-09-29, removed again the
  same day): GitHub's LFS API rejects any *single object* over **2 GiB** with
  `422 Size must be less than or equal to 2147483648`. This is a separate,
  per-object ceiling, not the 100 MB per-blob limit that LFS bypasses. Worse, the
  LFS **pre-push hook aborts the whole push**, and GitHub's **server-side**
  pre-receive hook independently refuses any push whose history references objects
  missing upstream — so `--no-verify` does not help either.
- Splitting the files into <2 GiB LFS parts would clear the object limit but
  33 GB exceeds the free Git LFS tier (10 GiB storage / 50 GiB bandwidth per
  month), so it would need LFS *billing*.

They are organiser files that ship with the starter pack, so nothing that cannot
be regenerated is missing from version control. `.gitignore` keeps
`datasets/train.parquet` and `datasets/valid.parquet` out; **do not re-add
`.gitattributes`** — the repository deliberately uses no LFS.

## Integrity

Verify a local copy against the organiser's original with:

```bash
shasum -a 256 datasets/train.parquet datasets/valid.parquet datasets/valid_mask.parquet
```

| File | sha256 |
|---|---|
| `train.parquet` | `4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81` |
| `valid.parquet` | `2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782` |
| `valid_mask.parquet` | `301430165527a6003d44486a8c5051288c31112d327df94fb420e82c018e5251` |

The last one is tracked in git, so a clone is already known-good.

## Regenerating the derived cache

Everything the models actually read is regenerable from the two large files in
about 7 minutes and 54 GB of disk — no model ever needs the parquets at
inference time:

```bash
python tools/make_shards.py --split both      # -> cache/train (46 GiB), cache/valid (8.2 GiB)
```

`tools/make_shards.py --split both --verify-only` re-checks an existing build
without touching it. The cache is gitignored.
