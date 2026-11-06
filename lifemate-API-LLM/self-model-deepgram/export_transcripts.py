"""Export ASR transcripts into single consolidated files for easy checking/reviewing."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'dataset/lifemate/train+public_test'
ASR_DIR = HERE / 'results/asr'
OUT_DIR = HERE / 'results'


def main():
    public_test_file = ROOT / 'public_test/public_test.jsonl'
    if not public_test_file.exists():
        raise FileNotFoundError(f"Missing {public_test_file}")

    rows = [json.loads(line) for line in public_test_file.read_text(encoding='utf-8').splitlines() if line.strip()]

    # Collect records in chronological conversation order
    items = []
    grouped_samples = []

    for row in rows:
        sample_id = row['id']
        sample_group = {
            'id': sample_id,
            'turns': []
        }

        # History user turns with audio
        history = row.get('history', [])
        u_idx = 1
        for turn in history:
            audio_path = turn.get('audio')
            if audio_path:
                stem = Path(audio_path).stem
                asr_file = ASR_DIR / f"{stem}.json"
                data = json.loads(asr_file.read_text(encoding='utf-8'))
                transcript = data.get('transcript', '')
                rec = {
                    'sample_id': sample_id,
                    'audio_path': audio_path,
                    'audio_stem': stem,
                    'turn_type': f"history_u{u_idx}",
                    'turn_label': f"turn {u_idx}",
                    'transcript': transcript,
                    'confidence': data.get('confidence')
                }
                items.append(rec)
                sample_group['turns'].append(rec)
                u_idx += 1

        # Final request audio
        final_audio = row.get('audio')
        if final_audio:
            stem = Path(final_audio).stem
            asr_file = ASR_DIR / f"{stem}.json"
            data = json.loads(asr_file.read_text(encoding='utf-8'))
            transcript = data.get('transcript', '')
            rec = {
                'sample_id': sample_id,
                'audio_path': final_audio,
                'audio_stem': stem,
                'turn_type': "final_request",
                'turn_label': "final",
                'transcript': transcript,
                'confidence': data.get('confidence')
            }
            items.append(rec)
            sample_group['turns'].append(rec)

        grouped_samples.append(sample_group)

    print(f"Total processed audio items: {len(items)}")
    assert len(items) == 1933, f"Expected 1933 items, got {len(items)}"

    # 1. Main transcripts.txt (<stem>: <transcript>)
    transcripts_txt = OUT_DIR / 'transcripts.txt'
    lines_txt = [f"{item['audio_stem']}: {item['transcript']}" for item in items]
    transcripts_txt.write_text('\n'.join(lines_txt) + '\n', encoding='utf-8')
    print(f"Written: {transcripts_txt} ({len(lines_txt)} lines)")

    # Also write directly inside ASR_DIR for convenience
    asr_transcripts_txt = ASR_DIR / 'transcripts.txt'
    asr_transcripts_txt.write_text('\n'.join(lines_txt) + '\n', encoding='utf-8')
    print(f"Written: {asr_transcripts_txt} ({len(lines_txt)} lines)")

    # 2. transcripts_grouped.txt (organized with sample ID headers)
    transcripts_grouped_txt = OUT_DIR / 'transcripts_grouped.txt'
    grouped_lines = []
    for g in grouped_samples:
        grouped_lines.append(f"[{g['id']}]")
        for t in g['turns']:
            grouped_lines.append(f"  [{t['turn_label']}] {t['audio_stem']}.wav: {t['transcript']}")
        grouped_lines.append("")
    transcripts_grouped_txt.write_text('\n'.join(grouped_lines), encoding='utf-8')
    print(f"Written: {transcripts_grouped_txt}")

    # 3. transcripts.tsv (tab-separated for Excel / Sheet viewer)
    transcripts_tsv = OUT_DIR / 'transcripts.tsv'
    tsv_lines = ["sample_id\taudio_stem\taudio_path\tturn_type\tconfidence\ttranscript"]
    for item in items:
        conf_str = f"{item['confidence']:.4f}" if item['confidence'] is not None else ""
        clean_text = item['transcript'].replace('\t', ' ').replace('\n', ' ')
        tsv_lines.append(f"{item['sample_id']}\t{item['audio_stem']}\t{item['audio_path']}\t{item['turn_type']}\t{conf_str}\t{clean_text}")
    transcripts_tsv.write_text('\n'.join(tsv_lines) + '\n', encoding='utf-8')
    print(f"Written: {transcripts_tsv}")

    # 4. transcripts_raw.txt (only the transcript text, 1 per line)
    transcripts_raw_txt = OUT_DIR / 'transcripts_raw.txt'
    raw_lines = [item['transcript'] for item in items]
    transcripts_raw_txt.write_text('\n'.join(raw_lines) + '\n', encoding='utf-8')
    print(f"Written: {transcripts_raw_txt} ({len(raw_lines)} lines)")

    # 5. transcripts.jsonl (standard JSONL format by sample)
    transcripts_jsonl = OUT_DIR / 'transcripts.jsonl'
    with transcripts_jsonl.open('w', encoding='utf-8') as f:
        for row in rows:
            sid = row['id']
            main_audio = row['audio']
            main_stem = Path(main_audio).stem
            main_data = json.loads((ASR_DIR / f"{main_stem}.json").read_text(encoding='utf-8'))
            history_data = []
            for h in row.get('history', []):
                h_audio = h.get('audio')
                if h_audio:
                    h_stem = Path(h_audio).stem
                    h_asr = json.loads((ASR_DIR / f"{h_stem}.json").read_text(encoding='utf-8'))
                    history_data.append({
                        'audio': h_audio,
                        'text': h_asr.get('transcript', ''),
                        'confidence': h_asr.get('confidence')
                    })
            record = {
                'id': sid,
                'audio': main_audio,
                'text': main_data.get('transcript', ''),
                'confidence': main_data.get('confidence'),
                'history': history_data
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Written: {transcripts_jsonl}")


if __name__ == '__main__':
    main()
