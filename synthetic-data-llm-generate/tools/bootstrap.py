"""Copy only official tool schema/runtime; never copy test questions or states."""
import argparse
import os
import sys
from pathlib import Path
from pipeline import DEFAULT_OUT, PipelineError, ROOT, atomic_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT.parent / 'dataset/lifemate/train+public_test/public_test')
    ap.add_argument('--out', type=Path, default=Path(os.environ.get('SYN_OUT', DEFAULT_OUT)))
    a = ap.parse_args()
    catalog = a.source / 'public_test_tools_catalog.json'
    runtime = a.source / 'zaic_tools'
    if not catalog.is_file() or not (runtime / 'executor.py').is_file():
        raise PipelineError('source cần public_test_tools_catalog.json và zaic_tools/executor.py')
    files = [(catalog, a.out / 'train_tools_catalog.json')]
    files += [(p, a.out / 'zaic_tools' / p.relative_to(runtime)) for p in runtime.rglob('*.py')]
    # Preflight before any write; simulator edits invalidate all previous reviews.
    for src, dst in files:
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise PipelineError(f'{dst} khác bản BTC; giữ nguyên, cần kiểm tra thủ công')
    for src, dst in files:
        if not dst.exists():
            atomic_bytes(dst, src.read_bytes())
    print(f'OK: schema/runtime ({len(files)} files); không đọc episode hoặc environment test')


if __name__ == '__main__':
    try:
        main()
    except (PipelineError, OSError, ValueError) as e:
        sys.exit(f'ERR: {e}')
