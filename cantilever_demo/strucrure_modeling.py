# strucrure_modeling.py
# -----------------------------------------------------------------------------
#  Cantilever Pier Modeling & OpenSeesPy Time History Analysis
#
#  Authors: Jianwei Zhang, Chao Yu, Xiya Liu
#  Affiliation: College of Civil Engineering, Tongji University
#
#  Copyright © 2025 Jianwei Zhang, Chao Yu, Xiya Liu. All rights reserved.
# -----------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import Queue
from typing import Callable
import openseespy.opensees as ops  # type: ignore[import]

from paramters_sampling import PierParams
from utils.Record import AccRecord, GRAVITY
from utils.progress import ProgressMessage, send_progress_message

ResultRow = dict[str, float | int | str]


def _safe_send_progress(
    queue: Queue[ProgressMessage] | None,
    worker_name: str | None,
    stage: str,
    step: int,
    total: int,
) -> None:
    if queue is None or worker_name is None:
        return
    send_progress_message(queue, worker_name, stage, step, total)


def run_one_earthquake(
    params: PierParams,
    record: AccRecord,
    *,
    stage_callback: Callable[[str, int, int], None] | None = None,
    stage_index: int = 1,
    stage_total: int = 1,
) -> tuple[float, float, float]:
    """Modeling -> Gravity Static Analysis -> Seismic Time History Analysis"""
    ops.wipe()
    try:
        # ------------------------------------------------------------------
        # Modeling: 1m element discretization, using dispBeamColumn with circular fiber section
        # ------------------------------------------------------------------
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        n_elem = max(1, int(round(params.h)))
        dz = params.h / n_elem

        for i in range(n_elem + 1):
            ops.node(i + 1, 0.0, i * dz)

        base_node = 1
        top_node = n_elem + 1
        ops.fix(base_node, 1, 1, 1)
        ops.mass(top_node, params.m_top, params.m_top, 0.0)

        if stage_callback is not None:
            stage_callback("Pier Modeling", stage_index, stage_total)

        E_concrete = 30.0e6  # kN/m²
        mat_tag = 1
        sec_tag = 1
        transf_tag = 1
        integ_tag = 1

        ops.uniaxialMaterial("Elastic", mat_tag, E_concrete)
        ops.section("Fiber", sec_tag)
        ops.patch("circ", mat_tag, 32, 4, 0.0, 0.0, 0.0, params.d / 2.0)

        ops.geomTransf("PDelta", transf_tag)
        ops.beamIntegration("Lobatto", integ_tag, sec_tag, 2)

        for ele in range(1, n_elem + 1):
            ops.element("dispBeamColumn", ele, ele, ele + 1, transf_tag, integ_tag)

        # ------------------------------------------------------------------
        # Gravity Static Analysis
        # ------------------------------------------------------------------
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(top_node, 0.0, -params.m_top * GRAVITY, 0.0)

        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1.0e-8, 50)
        ops.algorithm("Newton")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        if ops.analyze(1) != 0:
            raise RuntimeError(
                f"Gravity analysis failed to converge: sample h={params.h:.2f} m, d={params.d:.2f} m, m_top={params.m_top:.1f} t"
            )

        static_dx = float(ops.nodeDisp(top_node, 1))
        static_dy = float(ops.nodeDisp(top_node, 2))

        if stage_callback is not None:
            stage_callback("Gravity Analysis", stage_index, stage_total)

        ops.loadConst("-time", 0.0)
        ops.remove("pattern", 1)
        ops.remove("timeSeries", 1)
        ops.wipeAnalysis()

        # ------------------------------------------------------------------
        # Seismic Time History Analysis
        # ------------------------------------------------------------------
        acc_values = record.acc_series
        if not acc_values:
            raise ValueError("Ground motion record is empty, cannot perform time history analysis")

        dt = record.dt
        ops.timeSeries("Path", 101, "-dt", dt, "-values", *acc_values)
        ops.pattern("UniformExcitation", 101, 1, "-accel", 101)

        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1.0e-6, 20)
        ops.algorithm("Newton")
        ops.integrator("Newmark", 0.5, 0.25)
        ops.analysis("Transient")

        pending_step = max(stage_index - 1, 0)
        total_steps = record.n_steps
        if stage_callback is not None:
            stage_callback("Seismic Analysis", pending_step, total_steps)

        import time
        total_steps = len(acc_values)
        max_disp = 0.0
        next_second = 0.1
        current_t = time.time()
        for step, _ in enumerate(acc_values, start=1):
            if ops.analyze(1, dt) != 0:
                raise RuntimeError(
                    f"Seismic analysis failed to converge: sample h={params.h:.2f} m, d={params.d:.2f} m, m_top={params.m_top:.1f} t, record={record.id}, step={step}"
                )

            disp = float(ops.nodeDisp(top_node, 1))
            max_disp = max(max_disp, abs(disp))

            if stage_callback is not None:
                if time.time() - current_t >= next_second:
                    stage_callback("Seismic Analysis", step, total_steps)
                    current_t = time.time()

        ops.wipeAnalysis()
        ops.remove("pattern", 101)
        ops.remove("timeSeries", 101)

        if stage_callback is not None:
            stage_callback("Seismic Analysis", stage_index, total_steps)

        return max_disp, static_dx, static_dy
    except Exception as exc:  # pylint: disable=broad-except
        if stage_callback is not None:
            stage_callback("Seismic Analysis", stage_index, stage_total)
        print(
            f"[Error] Seismic analysis exception: sample h={params.h:.2f} m, d={params.d:.2f} m, m_top={params.m_top:.1f} t, record={record.id}, error={exc}"
        )
        return float("nan"), float("nan"), float("nan")
    finally:
        ops.wipe()


def analyze_pier_with_records(
    params: PierParams,
    records: Sequence[AccRecord],
    *,
    sample_id: int,
    progress_queue: Queue[ProgressMessage] | None = None,
    worker_name: str | None = None,
) -> list[ResultRow]:
    """
    Performs time history analysis for multiple ground motion records 
    on the same pier parameter set sequentially.
    """
    results: list[ResultRow] = []
    gm_total = len(records)

    def _stage_callback(stage: str, step: int, total: int) -> None:
        _safe_send_progress(progress_queue, worker_name, stage, step, total)

    if gm_total == 0:
        return results

    for idx, record in enumerate(records, start=1):
        max_disp, _, _ = run_one_earthquake(
            params,
            record,
            stage_callback=_stage_callback,
            stage_index=idx,
            stage_total=gm_total,
        )

        results.append(
            {
                "sample_id": sample_id,
                "gm_id": record.id,
                "h_m": params.h,
                "d_m": params.d,
                "m_top_t": params.m_top,
                "max_disp_top_m": max_disp,
            }
        )

    _stage_callback("complete", 0, 0)
    return results