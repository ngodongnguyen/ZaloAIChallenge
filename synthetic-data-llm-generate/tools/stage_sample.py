"""Compute history/labels through simulator and atomically save a complete draft."""
import argparse
import json
import sys
from pipeline import Pipeline, PipelineError, read_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group', required=True)
    ap.add_argument('--input', required=True, help='proposal JSON; see prompt_generate.md')
    ap.add_argument('--replace', action='store_true', help='replace this ID in the same group; invalidates old reviews')
    a = ap.parse_args()
    row = Pipeline().stage(a.group, read_json(a.input), replace=a.replace)
    print(json.dumps({'id': row['id'], 'status': 'draft', 'n_calls': len(row['label']['tool_calls'])}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (PipelineError, OSError, KeyError, TypeError, ValueError) as e:
        sys.exit(f'ERR: {e}')
