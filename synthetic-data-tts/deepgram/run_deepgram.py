"""Run Deepgram Flux Hoa/Minh; reads local .env without displaying the key."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tts_common import cli

if __name__ == "__main__":
    args = sys.argv[1:]
    command = ["deepgram-voices"] if args == ["--list-voices"] else ["deepgram", *args]
    raise SystemExit(cli(command))
