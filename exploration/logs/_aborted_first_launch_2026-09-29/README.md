# Aborted first launch — 2026-09-29

These four logs are from the first attempt to train `gru_v001`–`gru_v004`.
Every run died ~30 s in, with **no traceback**: the shell that launched them was
aborted, and `nohup ... &` does not detach a process from its parent's process
group, so the group kill took the training processes with it. Two of them
(`v001`, `v002`) additionally contain a short 100-second dry run used to verify
that `max_steps` is correctly derived from `epochs`.

No checkpoints were ever written by any of these runs.

Kept for the record; the versions were reset to `status: planned` and relaunched
using `tools/launch.py`, which calls `os.setsid()` to escape the process group.
That mechanism was verified by killing a supervisor's entire process group with
SIGKILL and confirming the detached child kept running.
