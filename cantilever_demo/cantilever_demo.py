# cantilever_pier_lhs_openseespy.py
# -----------------------------------------------------------------------------
#  Cantilever Pier + LHS Parameter Sampling + OpenSeesPy Time History Analysis
#
#  Authors: Jianwei Zhang, Chao Yu, Xiya Liu
#  Affiliation: College of Civil Engineering, Tongji University
#
#  Copyright © 2025 Jianwei Zhang, Chao Yu, Xiya Liu. All rights reserved.
# -----------------------------------------------------------------------------
#
#  Features:
#    1. Defines the pier parameter model (h, d, m_top) using Pydantic.
#    2. Generates a batch of parameter samples using Latin Hypercube Sampling (LHS).
#    3. Builds a cantilever pier model and performs time history analysis for 
#       each sample across all ground motion records.
#    4. Records the maximum lateral displacement (EDP) at the top node and 
#       saves the results to a CSV file.
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import multiprocessing as mp
from collections.abc import Sequence
from functools import partial
from multiprocessing import Queue
from pathlib import Path

from paramters_sampling import PierParams, generate_pier_samples
from strucrure_modeling import ResultRow, analyze_pier_with_records
from utils.Record import AccRecord
from utils.progress import ProgressMessage, progress_map, send_progress_message


# =============================================================================
# 0. Global Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
GM_FOLDER = BASE_DIR / "waves"
GM_IDS: tuple[int, ...] = (1, 2, 3)
N_SAMPLES: int = 30
RESULT_CSV = BASE_DIR / "pier_lhs_results.csv"
MAX_CONCURRENCY = max(1, min(4, mp.cpu_count() or 1))   # Default to 4 processes


# =============================================================================
# 1. Parallel Task Definition
# =============================================================================

def _sample_worker(
    sample_index: int,
    params: PierParams,
    *,
    gm_ids: Sequence[int],
    gm_folder: Path,
    results_queue: Queue[list[ResultRow]],
    progress_queue: Queue[ProgressMessage] | None = None,
    worker_name: str | None = None,
) -> None:
    """
    Task executed by each process: runs the full ground motion analysis 
    for a single parameter sample.
    """
    sample_results: list[ResultRow] = []
    try:
        records = [AccRecord.get_acc(gm_folder, id=gm_id) for gm_id in gm_ids]
        sample_results = analyze_pier_with_records(
            params,
            records,
            sample_id=sample_index,
            progress_queue=progress_queue,
            worker_name=worker_name,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[Error] Analysis failed for sample #{sample_index}: {exc}")
        for gm_id in gm_ids:
            sample_results.append(
                {
                    "sample_id": sample_index,
                    "gm_id": gm_id,
                    "h_m": params.h,
                    "d_m": params.d,
                    "m_top_t": params.m_top,
                    "max_disp_top_m": float("nan"),
                }
            )
        if progress_queue is not None and worker_name is not None:
            send_progress_message(progress_queue, worker_name, "complete", 0, 0)
    finally:
        results_queue.put(sample_results)


def run_parametric_study(samples: Sequence[PierParams]) -> list[ResultRow]:
    """
    Executes ground motion analysis for all samples and displays 
    execution progress using multiprocessing.
    """
    if not samples:
        return []

    results_queue: Queue[list[ResultRow]] = mp.Queue()
    worker = partial(
        _sample_worker,
        gm_ids=GM_IDS,
        gm_folder=GM_FOLDER,
        results_queue=results_queue,
    )

    task_args = [(idx, sample) for idx, sample in enumerate(samples, start=1)]

    stage_def = [("Modeling", 1), ("Gravity Analysis", len(GM_IDS)), ("Seismic Analysis", len(GM_IDS))]
    progress_map(
        worker,
        task_args,
        description="Parametric Analysis",
        max_concurrency=min(MAX_CONCURRENCY, len(task_args)),
        stage_factory=stage_def,
        task_name_getter=lambda args, _: f"Sample#{args[0]:03d}",
    )

    results: list[ResultRow] = []
    try:
        for _ in task_args:
            results.extend(results_queue.get())
    finally:
        results_queue.close()
        results_queue.join_thread()

    results.sort(key=lambda item: (item["sample_id"], item["gm_id"]))
    return results


# =============================================================================
# 2. Result Output
# =============================================================================

def save_results_csv(results: Sequence[ResultRow], filename: Path) -> None:
    """
    Saves the list of results to a CSV file.
    """
    if not results:
        print("[Info] Results are empty, no CSV generated.")
        return

    fieldnames = [
        "sample_id",
        "gm_id",
        "h_m",
        "d_m",
        "m_top_t",
        "max_disp_top_m",
    ]

    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            formatted_row = {
                key: (f"{value:.3f}" if isinstance(value, float) else value)
                for key, value in row.items()
                if key in fieldnames
            }
            writer.writerow(formatted_row)

    print(f"\n[Info] Results saved to CSV file: {filename}")


# =============================================================================
# 3. Main Program
# =============================================================================

def main() -> None:
    mp.freeze_support()

    print("=" * 60)
    print("Cantilever Pier LHS Sampling + OpenSeesPy Time History Analysis")
    print("=" * 60)

    samples = generate_pier_samples(N_SAMPLES, seed=42)
    print(f"\nGenerated {len(samples)} parameter samples:")
    for i, s in enumerate(samples, start=1):
        print(f"  Sample #{i}: h = {s.h:.2f} m, d = {s.d:.2f} m, m_top = {s.m_top:.1f} t, A = {s.A:.3f} m², Iz = {s.Iz:.4f} m⁴")

    print("\nStarting parallel parametric analysis, 3 ground motions per sample...")
    results = run_parametric_study(samples)

    save_results_csv(results, RESULT_CSV)

    print("\nResult Preview (floats rounded to 3 decimal places):")
    for row in results:
        pretty_row = {
            key: (f"{value:.3f}" if isinstance(value, float) else value)
            for key, value in row.items()
        }
        print(pretty_row)
    print("\n" + "=" * 60)
    print("Program finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()