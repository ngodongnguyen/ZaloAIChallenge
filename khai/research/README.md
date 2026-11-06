# research/

The Kaggle experiment harness behind the submissions from 49.71 on, plus the synthetic training data.
It is a snapshot copied from a local working folder (`C:/Project/zalo_research`); scripts still use that
absolute path and the local Python venv, so adjust `RESEARCH` in `harness/evalrun.py` and the `sys.path`
lines before running them elsewhere.

## Layout

| Path | What it is |
|---|---|
| `harness/` | Kaggle API wrapper (accounts are aliases `acc1`…`acc5`; each alias reads its username and token from `KAGGLE_USER_ACCn` / `KAGGLE_TOKEN_ACCn`, never from files), weekly GPU budget ledger, run registry, split loader, approximate local metric |
| `kernel_template/run_eval.py` | The Kaggle kernel: optional LoRA SFT on 2× T4, dev eval on a split, full public test → `submission.jsonl` |
| `kernel_template/run_asr_ft.py` | Kaggle kernel: LoRA fine-tune of PhoWhisper-large, dev WER before and after |
| `loop/sft_run.py` | Launch SFT + eval (`--e16`, `--final`, `--synth --synth-dir`, `--split`, `--no-test`, ...) |
| `loop/infer_run.py` | Inference only with existing adapters (`--asr-run`, `--gold-text`, `--split`) |
| `loop/asr_ft_run.py` | Launch ASR fine-tuning (`--history` adds history-turn audio) |
| `loop/ceiling_run.py` | Runs the team's prompt and agent with a Codex model as the LLM, **train ids only**, to measure a ceiling |
| `splits/` | `lifemate_split.json` (dev ids + gold state hashes), `unseen_apps.json` (187 samples on Nguyên's 6 held-out apps, the main generalisation test), `smoke40.json`, `trained120.json`, Nguyên's holdout id lists |
| `synth/` | Synthetic data generator and data (below) |
| `tests/` | Unit tests for the metric, configs, registry |

## Synthetic data (`synth/`)

Generated with Codex CLI (`gpt-6.1-sol`) in an empty read-only folder. The prompts contain nothing from
train, public_test or private_test; apps are invented. The data is used for SFT only, never at inference.

| File | Content |
|---|---|
| `apps.json` | Round 1: 30 invented apps |
| `ok.jsonl` | Round 1: 672 scenarios that passed `validate.py` (used by r021, LB 51.45) |
| `v2/apps_all.json` | Round 2 catalog: round-1 apps + 27 new ones (57) |
| `v2/plan.json` | Round 2 plan: 140 batches × 14 scenarios, each with a focus (conditional, dates, pick_id, ask_or_act, self_correct, multi_step, history, exact_text, odd, general) |
| `v2/ok_v2.jsonl` | Round 2 scenarios that passed `validate.py` so far (generation in progress; ids are opaque, e.g. `pol_4812`) |

`validate.py` checks structure only (tool exists, argument types, ids match tool results), not whether the
label is semantically right. `to_items.py` turns scenarios into train-like items plus a merged catalog for
`train/build_sft.py --synth-dir`:

```
python synth/to_items.py --apps synth/v2/apps_all.json --scen <ok.jsonl + v2/ok_v2.jsonl> --out synth/out_v12
```

Known difference from real data: invented apps mostly use search tools with required arguments, while real
read tools are mostly argument-free "list everything, then pick an id" (83% of real read calls vs 5–8% synthetic).

## Findings that shaped the current recipe

- One seed alone moves the LB by ~1.4 (r018 51.64 vs r020 50.21); treat LB gaps under 2 as noise.
- On the 187 unseen-app samples, synthetic data helps (0.589 vs 0.538); renaming tools does not (0.545).
- r021 scores 0.862 on 120 samples it trained on (gold text) vs 0.589 on unseen apps: the gap is generalisation, not capacity.
- A strong API model with our prompt reaches 0.82 with gold text; its losses are labelling conventions (which optional arguments to pass, extra read calls), not reasoning.
- ASR LoRA a002 trained on 158 of the 187 unseen-app audio files, so local scores on that split overstate ASR quality. Retraining ASR with history clips (WER 0.134 → 0.110) gave no LB gain.
