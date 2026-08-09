"""Run training once while recording durable progress, completion, and errors."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_DIR = Path("outputs")
STATUS_PATH = OUTPUT_DIR / "training-status.json"
LOG_PATH = OUTPUT_DIR / "training.log"


def write_status(state: str, **details) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    payload = {
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preprocess-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--unlock-final-test", action="store_true")
    args = parser.parse_args()

    command = [sys.executable, "-u", "gemini-training.py"]
    if args.preprocess_only:
        command.append("--preprocess-only")
    elif args.smoke_test:
        command.append("--smoke-test")
    if args.unlock_final_test:
        command.append("--unlock-final-test")

    write_status("running", command=command, log=str(LOG_PATH))
    with LOG_PATH.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        write_status(
            "running",
            command=command,
            log=str(LOG_PATH),
            pid=process.pid,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()

    state = "completed" if return_code == 0 else "failed"
    write_status(
        state,
        command=command,
        log=str(LOG_PATH),
        return_code=return_code,
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
