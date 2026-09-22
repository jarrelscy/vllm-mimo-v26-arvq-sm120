# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Durable handoff from candidate fitting to allocation-aware hybrid PV."""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work, parent = args.work, args.parent
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / "handoff.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(stage, **values):
        tmp = work / "handoff_status.tmp"
        tmp.write_text(
            json.dumps({"stage": stage, "time": time.time(), **values}, indent=2)
        )
        tmp.replace(work / "handoff_status.json")

    status("waiting_for_candidate_campaign", target_hot_experts=1325)
    while not (parent / "receipts/full_model69.json").exists():
        time.sleep(30)
    # Keep fitting independent of uploader success or a stalled HF request.
    (parent / "STOP_PUBLISH").touch()
    publisher_lock = work / "publisher.lock"
    if not publisher_lock.exists():
        publisher_lock.symlink_to(parent / "publisher.lock")
    here = Path(__file__).resolve().parent
    status("allocation_and_hybrid_preparation")
    if not (work / "hybrid_prepared.json").exists():
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            str(here / "prepare_hybrid.py"),
            "--parent",
            str(parent),
            "--work",
            str(work),
        ]
        with (work / "prepare.log").open("a") as log:
            for attempt in range(3):
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 0:
                    break
                status("preparation_retry", attempt=attempt + 1, code=result.returncode)
            else:
                status("preparation_failed")
                raise RuntimeError("Hybrid preparation failed; see prepare.log")
    status("hybrid_pv_running")
    with (work / "driver.log").open("a") as log:
        result = subprocess.run(
            [sys.executable, "-u", str(here / "driver.py"), "--work", str(work)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    status("driver_exited", returncode=result.returncode)


if __name__ == "__main__":
    main()
