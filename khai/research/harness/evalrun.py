"""One evaluation on Kaggle: snapshot the workspace code, publish it as a dataset version, push the
eval kernel with this run's parameters embedded, wait, download the output, and score it.

Every step runs under one Kaggle account (ACCOUNTS). Each account has its own kernel, code dataset,
data dataset and budget ledger, so two accounts can run two evaluations side by side.
"""
import io
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from harness import kaggle_io
from harness.configs import config_hash, code_hash
from harness.score import score_run

RESEARCH = Path("C:/Project/zalo_research")
TEMPLATE = RESEARCH / "kernel_template" / "run_eval.py"


class _Accounts(dict):
    """Per-account Kaggle refs. ("acc1", "slug") becomes "<username of acc1>/slug" on lookup, so usernames
    stay in environment variables (see kaggle_io) and out of the code."""

    def __getitem__(self, account):
        spec = super().__getitem__(account)
        return {k: f"{kaggle_io.username(v[0])}/{v[1]}" if isinstance(v, tuple) else v for k, v in spec.items()}


ACCOUNTS = _Accounts({
    "acc1": {
        "code_dataset": ("acc1", "lifemate-code"),
        "data_dataset": ("acc1", "zaloai-lifemate-dataset"),
        "data_subdir": "/train+public_test",
        "kernel": ("acc1", "lifemate-eval"),
    },
    "acc2": {
        "code_dataset": ("acc2", "lifemate-code"),
        "data_dataset": ("acc2", "zaloai-lifemate-dataset"),
        "data_subdir": "",  # this upload was extracted without the train+public_test folder level
        "kernel": ("acc2", "lifemate-eval"),
    },
    # Nguyên's own account: his lifemate-code dataset and lifemate-train kernel live there, so ours use a
    # khai- prefix and must never touch his. The data is acc1's dataset, which his kernels already mount.
    "acc3": {
        "code_dataset": ("acc3", "khai-lifemate-code"),
        "data_dataset": ("acc1", "zaloai-lifemate-dataset"),
        "data_subdir": "/train+public_test",
        "kernel": ("acc3", "khai-lifemate-eval"),
    },
    "acc4": {
        "code_dataset": ("acc4", "khai-lifemate-code"),
        "data_dataset": ("acc4", "khai-lifemate-data"),
        "data_subdir": "",  # same upload as acc2
        "kernel": ("acc4", "khai-lifemate-eval"),
    },
    "acc5": {
        "code_dataset": ("acc5", "khai-lifemate-code"),
        "data_dataset": ("acc5", "khai-lifemate-data"),
        "data_subdir": "",  # same upload as acc2; created only when a run is assigned here
        "kernel": ("acc5", "khai-lifemate-eval"),
    },
})


def snapshot_workspace(workspace, dest, account="acc1", adapters=None):
    """Export the committed code (src/, train/, requirements.txt) from the workspace HEAD.

    `adapters` ({name: local dir}) are LoRA adapters copied to adapters/<name> for the kernel to merge."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(workspace), "archive", "HEAD", "src", "train", "requirements.txt"],
                             check=True, capture_output=True)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tf:
        tf.extractall(dest, filter="data")
    asr_cache = RESEARCH / "data" / "asr_train.jsonl"  # PhoWhisper transcripts of train audio, for SFT
    if asr_cache.exists():
        (dest / "data").mkdir()
        shutil.copy(asr_cache, dest / "data" / "asr_train.jsonl")
    for name, path in (adapters or {}).items():
        shutil.copytree(path, dest / "adapters" / name)
    with open(dest / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump({"title": ACCOUNTS[account]["code_dataset"].split("/")[1], "id": ACCOUNTS[account]["code_dataset"],
                   "licenses": [{"name": "unknown"}]}, f)
    return dest


def mount_paths(account):
    """Where Kaggle mounts the attached datasets for this account's kernels."""
    code_owner, code_slug = ACCOUNTS[account]["code_dataset"].split("/")
    data_owner, data_slug = ACCOUNTS[account]["data_dataset"].split("/")
    return (f"/kaggle/input/datasets/{code_owner}/{code_slug}",
            f"/kaggle/input/datasets/{data_owner}/{data_slug}{ACCOUNTS[account]['data_subdir']}")


def write_kernel(folder, params, account="acc1", template=TEMPLATE):
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    code_in, data_root = mount_paths(account)
    src = (Path(template).read_text(encoding="utf-8")
           # json.loads of a quoted string: bare JSON would put true/false/null in the script as Python names
           .replace("__PARAMS__", "json.loads(" + repr(json.dumps(params, ensure_ascii=False)) + ")")
           .replace("__CODE_IN__", code_in)
           .replace("__DATA_ROOT__", data_root))
    (folder / "run_eval.py").write_text(src, encoding="utf-8")  # kernel file name is fixed whatever the template
    meta = {
        # Kaggle derives a new kernel's slug from its title, so the title must equal the slug part of the id.
        "id": ACCOUNTS[account]["kernel"], "title": ACCOUNTS[account]["kernel"].split("/")[1],
        "code_file": "run_eval.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [ACCOUNTS[account]["code_dataset"], ACCOUNTS[account]["data_dataset"]],
        "competition_sources": [], "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def evaluate(label, params, split_path, workspace, registry, budget, est_s=3600,
             timeout_s=6 * 3600, cfg=None, account="acc1", adapters=None):
    """Run one eval under `account`. `cfg` is the config dict used for hashing (hypothesis and parent)."""
    if not budget.can_run(est_s):
        raise RuntimeError(f"budget line reached for {account}; not starting a Kaggle run")
    workdir = RESEARCH / "loop" / "runs_work" / label
    snap = snapshot_workspace(workspace, workdir / "snapshot", account, adapters)
    code_h = code_hash({f"{d}/{p.name}": p for d in ("src", "train") for p in sorted((snap / d).glob("*.py"))})
    cfg_h = config_hash(cfg) if cfg else "none"

    run_id = registry.new_run(cfg or {"hypothesis": "unspecified run", "parent": None}, cfg_h, code_h,
                              run_id=label)
    registry.set_status(run_id, "running", extra={"account": account})
    if kaggle_io.dataset_exists(ACCOUNTS[account]["code_dataset"], account=account):
        kaggle_io.new_dataset_version(snap, f"{label}", account=account)
    else:
        kaggle_io.new_dataset(snap, account=account)
    expected = {f"{d}/{p.name}": p.stat().st_size for d in ("src", "train") for p in (snap / d).glob("*.py")}
    if (snap / "data" / "asr_train.jsonl").exists():
        expected["data/asr_train.jsonl"] = (snap / "data" / "asr_train.jsonl").stat().st_size
    for w in (snap / "adapters").glob("*/adapter_model.safetensors"):
        expected[w.relative_to(snap).as_posix()] = w.stat().st_size
    kaggle_io.wait_dataset_files(ACCOUNTS[account]["code_dataset"], expected, account=account)
    kdir = write_kernel(workdir / "kernel", params, account)
    kaggle_io.push_kernel(kdir, account)
    start = time.time()
    return collect(label, params, split_path, registry, budget, since=start, timeout_s=timeout_s,
                   account=account)


def collect(label, params, split_path, registry, budget, since, timeout_s=6 * 3600, account="acc1"):
    """Wait for the eval kernel pushed for `label`, download its output, score it, record the run.

    `since` is the push time (or the original push time when resuming a run already on Kaggle)."""
    run_id = label
    workdir = RESEARCH / "loop" / "runs_work" / label
    kernel = ACCOUNTS[account]["kernel"]
    status, elapsed = kaggle_io.wait_kernel(kernel, timeout_s, since=since, account=account)
    budget.record(run_id, elapsed, since, time.time())
    if status != "COMPLETE":
        registry.set_status(run_id, "failed", stop_reason=f"kernel status {status} on {account}")
        raise RuntimeError(f"kernel finished with {status}")

    out_dir = workdir / "output"
    kaggle_io.download_kernel_output(kernel, out_dir, account=account)
    out_file = out_dir / "out" / f"{params['out_name']}.jsonl"
    metrics = score_run(out_file, split_path)
    workdir.mkdir(parents=True, exist_ok=True)
    with open(workdir / "per_sample.json", "w", encoding="utf-8") as f:
        json.dump(metrics.pop("per_sample"), f, ensure_ascii=False, indent=1)
    shutil.copy(out_file, workdir / "output.jsonl")
    metrics["kernel_seconds"] = elapsed
    metrics["account"] = account
    registry.write_metrics(run_id, metrics)
    registry.set_status(run_id, "done", extra={"account": account})
    return run_id, metrics
