# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resumable sequential initial-fit/PV campaign with an independent publisher."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "recipe"))
from arvq88.inputs import write  # noqa: E402
from arvq88.pack import export_layer  # noqa: E402
from hybrid import allocation  # noqa: E402


def export(work, layer, phase):
    _, cold = allocation(work, layer)
    initial = work / "baseline" / "initial" / f"layer_{layer:05d}"
    report = None
    if phase == "initial":
        source = initial
    else:
        fit = work / f"layer{layer}_same_input"
        report = json.loads((fit / "report.json").read_text())
        if not report["complete"]:
            raise ValueError("PV report incomplete")
        if (
            abs(
                report["serialized_validation"]["target_rel"]
                - report["final_validation"]["target_rel"]
            )
            > 2e-6
        ):
            raise ValueError("PV serialization failed")
        source = fit / "merged"
        source.mkdir(exist_ok=True)
        rejected = (
            report["development_audit"]["target_rel"]
            > report["initial_development_audit"]["target_rel"] + 2e-6
        )
        rejected |= (
            report["final_validation"]["target_rel"]
            > report["initial_validation"]["target_rel"] + 2e-6
        )
        if rejected:
            # The audit is a one-shot gate. Do not search checkpoints against it.
            for projection in ("w13", "w2"):
                shutil.copy2(initial / f"{projection}.pt", source / f"{projection}.pt")
            decision = {
                "selection": "initial",
                "reason": "PV validation/audit regression",
                "best_step": 0,
                "pv_attempt_report": report,
            }
        else:
            ranks = [
                torch.load(fit / f"experts_rank{rank}.pt", weights_only=True, mmap=True)
                for rank in range(8)
            ]
            for projection in ("w13", "w2"):
                merged = dict(ranks[0]["encoded"][projection])
                for key in ("c0", "c1", "a", "b", "s"):
                    exemplar = merged[key]
                    result = torch.empty(
                        (len(cold), *exemplar.shape[1:]), dtype=exemplar.dtype
                    )
                    for rank in ranks:
                        result[rank["slots"]] = rank["encoded"][projection][key]
                    merged[key] = result
                torch.save({projection: merged}, source / f"{projection}.pt")
            decision = {
                "selection": "pv",
                "best_step": report["best_step"],
                "report": report,
            }
        write(fit / "selection.json", decision)
        manifest = json.loads((initial / "arvq-manifest.json").read_text())
        manifest["fit"] = "PV attempted; selected " + decision["selection"]
        manifest["selection"] = decision["selection"]
        write(source / "arvq-manifest.json", manifest)
    destination = work / "exports" / phase / f"layer{layer}"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = export_layer(source, destination, layer)
    if report is not None:
        write(destination / "pv_report.json", report)
    write(
        destination / "ready.json",
        {"complete": True, "phase": phase, "layer": layer, "files": manifest["files"]},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    lock = (work / "driver.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = work / "logs"
    logs.mkdir(exist_ok=True)
    receipts = work / "receipts"
    receipts.mkdir(exist_ok=True)
    cfg = json.loads((work / "config.json").read_text())
    for gate in (
        "source_verified.json",
        "capture_attention_qualified.json",
        "corpus/evaluation_selection.json",
    ):
        if not (work / gate).exists():
            raise ValueError(f"Missing preparation gate: {gate}")
    qualification = json.loads((work / "capture_attention_qualified.json").read_text())
    if not qualification.get("router_fp32_bias_qualified"):
        raise ValueError("Router FP32 correction bias has not been qualified")
    publisher = None

    def ensure_publisher():
        nonlocal publisher
        if publisher is None or publisher.poll() is not None:
            handle = (logs / "publisher.log").open("a")
            publisher = subprocess.Popen(
                [sys.executable, str(HERE / "publish.py"), "--work", str(work)],
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            handle.close()

    def run(stage, layer, command):
        receipt = receipts / f"{stage}{layer}.json"
        if receipt.exists():
            return
        for attempt in range(3):
            current_command = list(command)
            if stage == "pv":
                checkpoints = sorted(
                    (work / f"layer{layer}_same_input" / "training_checkpoints").glob(
                        "step_*/complete.json"
                    )
                )
                if checkpoints:
                    current_command += ["--resume", str(checkpoints[-1].parent)]
            write(
                work / "pipeline_status.json",
                {
                    "stage": stage,
                    "layer": layer,
                    "attempt": attempt + 1,
                    "time": time.time(),
                },
            )
            env = dict(os.environ, OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
            # Do not change capture batch geometry across resume: filenames and
            # sequence identities must remain stable for propagation.
            with (logs / f"{stage}{layer}.log").open("a") as handle:
                process = subprocess.Popen(
                    current_command, stdout=handle, stderr=subprocess.STDOUT, env=env
                )
                while process.poll() is None:
                    ensure_publisher()
                    time.sleep(10)
            if process.returncode == 0:
                write(
                    receipt, {"complete": True, "command": command, "time": time.time()}
                )
                return
            write(
                work / "last_failure.json",
                {
                    "stage": stage,
                    "layer": layer,
                    "returncode": process.returncode,
                    "attempt": attempt + 1,
                },
            )
            time.sleep(10)
        raise RuntimeError(
            f"Stage failed after three attempts: {stage}{layer}; see logs"
        )

    torchrun = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
    ]
    ensure_publisher()
    for layer in range(1, 70):
        if (work / "STOP").exists():
            write(
                work / "pipeline_status.json", {"stage": "stopped", "next_layer": layer}
            )
            return
        run(
            "capture",
            layer,
            torchrun
            + [
                str(HERE / "capture.py"),
                "--work",
                str(work),
                "--layer",
                str(layer),
                "--mode",
                "capture",
            ],
        )
        run(
            "initial",
            layer,
            torchrun
            + [
                str(HERE / "initial_fit.py"),
                "--work",
                str(work),
                "--layer",
                str(layer),
            ],
        )
        if not (work / "exports" / "initial" / f"layer{layer}" / "ready.json").exists():
            export(work, layer, "initial")
        run(
            "pv",
            layer,
            torchrun
            + [
                str(HERE / "recipe/arvq88/perf/sequential_pv_full_corpus.py"),
                "--work",
                str(work),
                "--layer",
                str(layer),
                "--target",
                "same_input",
                "--batch-tokens",
                str(cfg["batch_tokens"]),
                "--microbatch",
                str(cfg["microbatch_tokens"]),
                "--codebook-lr",
                str(cfg["codebook_lr"]),
                "--scale-lr",
                str(cfg["scale_lr"]),
                "--reassign-every",
                str(cfg["reassign_every"]),
            ],
        )
        if not (work / "exports" / "pv" / f"layer{layer}" / "ready.json").exists():
            export(work, layer, "pv")
        if layer < 69:
            run(
                "propagate",
                layer,
                torchrun
                + [
                    str(HERE / "capture.py"),
                    "--work",
                    str(work),
                    "--layer",
                    str(layer),
                    "--mode",
                    "propagate",
                ],
            )
        write(
            work / "pipeline_status.json",
            {"stage": "layer_complete", "layer": layer, "time": time.time()},
        )
        # Keep a one-layer recovery window; never delete source, corpus or fits.
        if layer > 1:
            for folder in (f"training_capture{layer - 1}", f"states{layer - 1}"):
                path = work / folder
                if path.exists():
                    shutil.rmtree(path)
    run(
        "full_model",
        69,
        torchrun + [str(HERE / "evaluate_model.py"), "--work", str(work)],
    )
    write(
        work / "pipeline_status.json",
        {
            "stage": "fit_complete",
            "layers": 69,
            "production_ready": False,
            "remaining": "review full-model quality and qualify SM120 serving",
        },
    )
    while publisher.poll() is None:
        time.sleep(30)


if __name__ == "__main__":
    main()
