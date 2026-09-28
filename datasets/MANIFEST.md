# Dataset manifest

The two large parquet files are **not** in git, and cannot be.

## Why

| File | Size | In git? | Reason |
|---|---:|---|---|
| `datasets/train.parquet` | 29.2 GB | no | GitHub LFS rejects any single object > **2 GiB** (`422 Size must be less than or equal to 2147483648`) |
| `datasets/valid.parquet` | 5.6 GB | no | same |
| `datasets/valid_mask.parquet` | 92 MB | yes (LFS) | under the limit, so it uploads normally |

They arrived with the organiser's starter pack and are byte-identical to it. They
are also fully re-obtainable by re-downloading the starter pack, so nothing that
cannot be regenerated is missing from version control.

Note this is **not** the usual 100 MB per-blob limit — that is bypassed by LFS
entirely. The 2 GiB ceiling is a separate, per-LFS-object limit that applies
regardless of splitting the repository or enabling billing. Splitting the files
into <2 GB parts would work around the object limit but would still need Git LFS
*billing*: the free tier allows 10 GiB of storage per month and these files total
33 GB, so the upload would be rejected on quota as well.

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

(The last one is also the LFS object id, so `git lfs fsck` covers it.)

## Regenerating the derived cache

Everything the models actually read is regenerable from the two large files in
about 7 minutes and 54 GB of disk — no model ever needs the parquets at
inference time:

```bash
python tools/make_shards.py --split both      # -> cache/train (46 GiB), cache/valid (8.2 GiB)
```

`tools/make_shards.py --split both --verify-only` re-checks an existing build
without touching it. The cache is gitignored.
