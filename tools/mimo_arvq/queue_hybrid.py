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

    status("waiting_for_current_layer_boundary", target_hot_experts=1325)
    (parent / "STOP").touch()
    with (parent / "driver.lock").open("a") as parent_lock:
        while True:
            try:
                fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(30)
    # Keep fitting independent of uploader success or a stalled HF request.
    (parent / "STOP_PUBLISH").touch()
    publisher_lock = work / "publisher.lock"
    if not publisher_lock.exists():
        publisher_lock.symlink_to(parent / "publisher.lock")
    here = Path(__file__).resolve().parent
    torchrun = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
    ]

    def run_task(stage, command, **values):
        status(stage, **values)
        with (work / f"{stage}.log").open("a") as log:
            for attempt in range(3):
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 0:
                    return
                status(stage + "_retry", attempt=attempt + 1, **values)
            status(stage + "_failed", **values)
            raise RuntimeError(f"{stage} failed; see its log")

    calibration = work / "allocation_calibration"
    if not (calibration / "allocation_capture_complete.json").exists():
        run_task(
            "small_allocation_capture",
            torchrun
            + [
                str(here / "allocation_capture.py"),
                "--parent",
                str(parent),
                "--work",
                str(calibration),
            ],
        )
    for layer in range(1, 70):
        candidates = (
            parent / f"layer{layer}_same_input/merged/arvq-manifest.json",
            parent / "baseline/initial" / f"layer_{layer:05d}/arvq-manifest.json",
            calibration / "baseline/initial" / f"layer_{layer:05d}/arvq-manifest.json",
        )
        if any(path.exists() for path in candidates):
            continue
        run_task(
            "lightweight_candidate_fit",
            torchrun
            + [
                str(here / "initial_fit.py"),
                "--work",
                str(calibration),
                "--layer",
                str(layer),
            ],
            layer=layer,
        )
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
