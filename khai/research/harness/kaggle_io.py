"""Kaggle access through the official client (no HTML scraping).

Accounts are named by aliases (acc1 ... acc5). Each alias's Kaggle username and API token are read from
environment variables at call time and never written to disk or to this repo:
  KAGGLE_USER_ACC1, KAGGLE_TOKEN_ACC1   for alias acc1 (and likewise for acc2 ... acc5)
The CLI's legacy push path rejects token auth, so kernel and dataset calls go through the SDK client,
which reads KAGGLE_API_TOKEN. This module sets that variable right before building each client.
"""
import os
import time

ACCOUNT_ALIASES = ("acc1", "acc2", "acc3", "acc4", "acc5")


def _env(name, account):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set in this process (needed for account {account})")
    return value


def username(account):
    return _env(f"KAGGLE_USER_{account.upper()}", account)


def _token(account):
    return _env(f"KAGGLE_TOKEN_{account.upper()}", account)


def _api(account="acc1"):
    token = _token(account)
    os.environ["KAGGLE_API_TOKEN"] = token
    os.environ["KAGGLE_USERNAME"] = username(account)
    os.environ["KAGGLE_KEY"] = token
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk import KaggleClient, KaggleEnv
    KaggleApi.build_kaggle_client = lambda self: KaggleClient(env=KaggleEnv.PROD, verbose=False)
    api = KaggleApi()
    api.authenticate()
    return api


def push_kernel(folder, account="acc1"):
    return _api(account).kernels_push(str(folder))


def _status_text(resp):
    """Plain state name such as 'RUNNING' or 'COMPLETE'. The SDK returns an enum like
    KernelWorkerStatus.RUNNING, so take the member name rather than str() of the enum."""
    value = getattr(resp, "status", None)
    if value is None and isinstance(resp, dict):
        value = resp.get("status")
    if hasattr(value, "name"):
        return value.name
    return str(value).rsplit(".", 1)[-1]


def kernel_status(ref, account="acc1"):
    return _status_text(_api(account).kernels_status(ref))


def wait_kernel(ref, timeout_s, poll_s=60, since=None, account="acc1"):
    """Poll until the kernel leaves RUNNING/QUEUED. Returns (final_status, elapsed_seconds).

    `since` lets a resumed collection count elapsed time from the original push."""
    api = _api(account)
    start = since or time.time()
    status = None
    failures = 0
    while True:  # always polls once: a resumed collection can start past the timeout
        try:
            status = _status_text(api.kernels_status(ref))
            failures = 0
        except Exception as e:  # transient network errors must not end the wait
            failures += 1
            print(f"status poll failed ({failures}): {e!r}", flush=True)
            if failures >= 30:
                raise
            time.sleep(poll_s)
            continue
        if status not in ("RUNNING", "QUEUED", "CANCEL_ACKNOWLEDGED") or time.time() - start >= timeout_s:
            break
        time.sleep(poll_s)
    return status, round(time.time() - start, 1)


def dataset_exists(ref, account="acc1"):
    owner = ref.split("/")[0]
    api, page = _api(account), 1
    while True:  # the search index lags behind new datasets, so list the owner's datasets instead
        batch = api.dataset_list(user=owner, page=page)
        if not batch:
            return False
        if any(d.ref == ref for d in batch):
            return True
        page += 1


def new_dataset(folder, account="acc1", dir_mode="zip"):
    return _api(account).dataset_create_new(str(folder), convert_to_csv=False, dir_mode=dir_mode)


def new_dataset_version(folder, notes, account="acc1"):
    return _api(account).dataset_create_version(str(folder), notes, convert_to_csv=False, dir_mode="zip")


def download_kernel_output(ref, dest, account="acc1"):
    return _api(account).kernels_output(ref, path=str(dest))


def wait_dataset_files(ref, expected, account="acc1", timeout_s=900, poll_s=15, settle_s=45):
    """Block until the dataset lists every file in `expected` ({path: size}) with the same size.

    A new dataset version is processed asynchronously; a kernel pushed before that finishes mounts the
    previous version and silently runs old code. `settle_s` adds a short pause after the sizes match."""
    api = _api(account)
    start = time.time()
    while time.time() - start < timeout_s:
        names, tok = {}, None
        while True:
            try:
                r = (api.dataset_list_files(ref, page_token=tok, page_size=200) if tok
                     else api.dataset_list_files(ref, page_size=200))
            except Exception as e:  # a just-created dataset answers 403 until it is processed
                print(f"waiting for dataset {ref}: {e}", flush=True)
                r = None
                break
            names.update({f.name: f.total_bytes for f in r.files})
            tok = getattr(r, "next_page_token", None)
            if not tok:
                break
        missing = [k for k, v in expected.items() if names.get(k) != v]
        if not missing:
            time.sleep(settle_s)
            return True
        print(f"waiting for dataset {ref}: {len(missing)} file(s) not yet current, e.g. {missing[:2]}", flush=True)
        time.sleep(poll_s)
    raise RuntimeError(f"dataset {ref} did not show the new code within {timeout_s}s")
