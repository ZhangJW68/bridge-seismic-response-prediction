# bridge_seismic_lhs_openseespy.py
# -----------------------------------------------------------------------------
#  Three-Span Continuous Beam Bridge + LHS Sampling + OpenSeesPy Seismic
#  Parametric Analysis
#
#  Authors: Jianwei Zhang, Chao Yu, Xiya Liu
#  Affiliation: College of Civil Engineering, Tongji University
#
#  Copyright © 2025 Jianwei Zhang, Chao Yu, Xiya Liu. All rights reserved.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, ClassVar, Callable, Union

import numpy as np
import numpy.random as npr
import openseespy.opensees as ops
from pydantic import BaseModel, Field, computed_field, ConfigDict
import matplotlib.pyplot as plt

# =============================================================================
# 1. Bridge Parameter Model (Pydantic)
# =============================================================================

class BridgeParams(BaseModel):
    """
    Pydantic model for bridge parameters.

    Core input parameters:
        span_length   : Span length (m)
        pier_height   : Pier height (m)
        pier_diameter : Pier diameter (m)
        deck_mass     : Deck mass per unit length (t/m)
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(validate_assignment=True)

    span_length: Annotated[
        float,
        Field(ge=20.0, le=60.0, description="Span length (m)")
    ]
    pier_height: Annotated[
        float,
        Field(ge=5.0, le=30.0, description="Pier height (m)")
    ]
    pier_diameter: Annotated[
        float,
        Field(ge=1.0, le=4.0, description="Pier diameter (m)")
    ]
    deck_mass: Annotated[
        float,
        Field(ge=5.0, le=20.0, description="Deck mass per unit length (t/m)")
    ]

    # Convenient cross-section properties (automatically computed from inputs)

    @computed_field(description="Pier cross-sectional area A (m²)")
    @property
    def A_pier(self) -> float:
        return math.pi * self.pier_diameter ** 2 / 4.0

    @computed_field(description="Pier moment of inertia Iz (m⁴)")
    @property
    def Iz_pier(self) -> float:
        return math.pi * self.pier_diameter ** 4 / 64.0


# =============================================================================
# 2. Latin Hypercube Sampling (LHS)
# =============================================================================

class LHSSampler:
    """
    Latin Hypercube Sampling utility (uniform distribution version only).
    """

    @staticmethod
    def lhs_uniform(
        n_samples: int,
        bounds: tuple[float, float],
        seed: int | None = None
    ) -> np.ndarray:
        """
        Performs 1D LHS within the given interval [lower, upper].
        """
        rng = npr.default_rng(seed)
        lower, upper = bounds

        # LHS over the [0, 1] interval
        u = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        rng.shuffle(u)

        # Scale to the specified interval
        return lower + (upper - lower) * u


def generate_bridge_samples(
    n_samples: int,
    span_bounds: tuple[float, float] = (30.0, 50.0),
    height_bounds: tuple[float, float] = (10.0, 20.0),
    diameter_bounds: tuple[float, float] = (1.5, 3.0),
    mass_bounds: tuple[float, float] = (8.0, 15.0),
    seed: int | None = 42,
) -> list[BridgeParams]:
    """
    Generates a list of bridge parameter samples using LHS.
    """
    base_seed = seed

    span_samples = LHSSampler.lhs_uniform(n_samples, span_bounds, seed=base_seed)
    height_samples = LHSSampler.lhs_uniform(
        n_samples, height_bounds, seed=None if base_seed is None else base_seed + 1
    )
    diameter_samples = LHSSampler.lhs_uniform(
        n_samples, diameter_bounds, seed=None if base_seed is None else base_seed + 2
    )
    mass_samples = LHSSampler.lhs_uniform(
        n_samples, mass_bounds, seed=None if base_seed is None else base_seed + 3
    )

    samples: list[BridgeParams] = []
    for i in range(n_samples):
        params = BridgeParams(
            span_length=float(span_samples[i]),
            pier_height=float(height_samples[i]),
            pier_diameter=float(diameter_samples[i]),
            deck_mass=float(mass_samples[i]),
        )
        samples.append(params)

    return samples


# =============================================================================
# 3. Ground Motion Record Processing
# =============================================================================

GRAVITY = 9.81  # m/s²

class AccRecord:
    """
    Ground motion acceleration record class.
    """
    def __init__(self, acc_series: list[float], dt: float, id: int):
        self.acc_series = acc_series
        self.dt = dt
        self.id = id
        self.n_steps = len(acc_series)

    @classmethod
    def get_acc(cls, folder: Path, id: int) -> 'AccRecord':
        """
        Reads the ground motion record with the specified ID from a folder.
        """
        filepath = folder / f"{id}.acc"

        if not filepath.exists():
            raise FileNotFoundError(f"Ground motion record file not found: {filepath}")

        with open(filepath, 'r') as f:
            lines = f.readlines()

        acc_data = []
        for line in lines:
            values = line.strip().split()
            for val in values:
                try:
                    acc_data.append(float(val))
                except ValueError:
                    continue

        # Assume a sampling frequency of 50 Hz (dt=0.02s); adjust as needed
        dt = 0.02
        return cls(acc_data, dt, id)


# =============================================================================
# 4. Bridge Analysis Functions
# =============================================================================

ResultRow = dict[str, Union[float, int, str]]


def run_one_earthquake(
    params: BridgeParams,
    record: AccRecord,
    *,
    stage_callback: Callable[[str, int, int], None] | None = None,
    stage_index: int = 1,
    stage_total: int = 1,
) -> tuple[float, float, float, list[float], list[float]]:
    """Modeling -> Gravity Static Analysis -> Seismic Time History Analysis"""
    ops.wipe()
    try:
        # ------------------------------------------------------------------
        # Modeling: build a three-span continuous beam bridge model
        # ------------------------------------------------------------------
        ops.model("basic", "-ndm", 2, "-ndf", 3)

        # Define node coordinates
        span = params.span_length
        h_pier = params.pier_height

        # Pier base nodes
        ops.node(1, 0.0, 0.0)       # Left abutment pier base
        ops.node(2, span, 0.0)      # Middle pier base
        ops.node(3, 2 * span, 0.0)  # Right abutment pier base

        # Pier top nodes
        ops.node(4, 0.0, h_pier)        # Left abutment pier top
        ops.node(5, span, h_pier)       # Middle pier top
        ops.node(6, 2 * span, h_pier)   # Right abutment pier top

        # Deck nodes
        ops.node(7, 0.0, h_pier + 1.0)              # Left deck node
        ops.node(8, span / 2, h_pier + 1.0)         # Midpoint of the first span
        ops.node(9, span, h_pier + 1.0)             # Middle deck node
        ops.node(10, span + span / 2, h_pier + 1.0) # Midpoint of the second span
        ops.node(11, 2 * span, h_pier + 1.0)        # Right deck node

        # Define materials
        E_concrete = 30.0e6  # kN/m² (30 GPa)
        ops.uniaxialMaterial("Elastic", 1, E_concrete)

        # Cross-section properties
        A_pier = params.A_pier
        Iz_pier = params.Iz_pier
        A_deck = 2.0    # Deck cross-sectional area (m²)
        Iz_deck = 0.67  # Deck moment of inertia (m⁴)

        # Define geometric transformation
        ops.geomTransf("Linear", 1)

        # Define pier elements
        ops.element('elasticBeamColumn', 1, 1, 4, A_pier, E_concrete, Iz_pier, 1)
        ops.element('elasticBeamColumn', 2, 2, 5, A_pier, E_concrete, Iz_pier, 1)
        ops.element('elasticBeamColumn', 3, 3, 6, A_pier, E_concrete, Iz_pier, 1)

        # Define deck-to-pier connection elements
        ops.element('elasticBeamColumn', 4, 4, 7, A_deck, E_concrete, Iz_deck, 1)
        ops.element('elasticBeamColumn', 5, 5, 9, A_deck, E_concrete, Iz_deck, 1)
        ops.element('elasticBeamColumn', 6, 6, 11, A_deck, E_concrete, Iz_deck, 1)

        # Define main girder elements
        ops.element('elasticBeamColumn', 7, 7, 8, A_deck, E_concrete, Iz_deck, 1)
        ops.element('elasticBeamColumn', 8, 8, 9, A_deck, E_concrete, Iz_deck, 1)
        ops.element('elasticBeamColumn', 9, 9, 10, A_deck, E_concrete, Iz_deck, 1)
        ops.element('elasticBeamColumn', 10, 10, 11, A_deck, E_concrete, Iz_deck, 1)

        # Define constraints
        ops.fix(1, 1, 1, 1)  # Fix left abutment pier base
        ops.fix(2, 1, 1, 1)  # Fix middle pier base
        ops.fix(3, 1, 1, 1)  # Fix right abutment pier base

        # Define mass
        mass_per_node = params.deck_mass * span / 2.0
        ops.mass(7, mass_per_node, mass_per_node, 0.0)
        ops.mass(8, mass_per_node, mass_per_node, 0.0)
        ops.mass(9, mass_per_node, mass_per_node, 0.0)
        ops.mass(10, mass_per_node, mass_per_node, 0.0)
        ops.mass(11, mass_per_node, mass_per_node, 0.0)

        if stage_callback is not None:
            stage_callback("Bridge Modeling", stage_index, stage_total)

        # ------------------------------------------------------------------
        # Gravity Static Analysis
        # ------------------------------------------------------------------
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)

        # Apply gravity load
        weight_per_length = params.deck_mass * GRAVITY

        ops.load(7, 0.0, -weight_per_length * span / 4.0, 0.0)
        ops.load(8, 0.0, -weight_per_length * span / 2.0, 0.0)
        ops.load(9, 0.0, -weight_per_length * span / 2.0, 0.0)
        ops.load(10, 0.0, -weight_per_length * span / 2.0, 0.0)
        ops.load(11, 0.0, -weight_per_length * span / 4.0, 0.0)

        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1.0e-8, 50)
        ops.algorithm("Newton")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        if ops.analyze(1) != 0:
            raise RuntimeError(
                f"Gravity analysis failed to converge: sample span={params.span_length:.2f} m, pier_h={params.pier_height:.2f} m, pier_d={params.pier_diameter:.2f} m"
            )

        static_dx = float(ops.nodeDisp(9, 1))
        static_dy = float(ops.nodeDisp(9, 2))

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
        ops.timeSeries("Path", 101, "-dt", dt, "-values", *acc_values, "-factor", GRAVITY)
        ops.pattern("UniformExcitation", 101, 1, "-accel", 101)

        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1.0e-6, 20)
        ops.algorithm("Newton")
        ops.integrator("Newmark", 0.5, 0.25)
        ops.analysis("Transient")

        total_steps = len(acc_values)
        max_disp = 0.0
        disp_history = []
        time_history = []

        for step, _ in enumerate(acc_values, start=1):
            if ops.analyze(1, dt) != 0:
                raise RuntimeError(
                    f"Seismic analysis failed to converge: sample span={params.span_length:.2f} m, pier_h={params.pier_height:.2f} m, pier_d={params.pier_diameter:.2f} m, record={record.id}, step={step}"
                )

            disp = float(ops.nodeDisp(9, 1))
            disp_history.append(disp)
            time_history.append(step * dt)
            max_disp = max(max_disp, abs(disp))

            if step % 100 == 0 and stage_callback is not None:
                stage_callback("Seismic Analysis", step, total_steps)

        ops.wipeAnalysis()
        ops.remove("pattern", 101)
        ops.remove("timeSeries", 101)

        if stage_callback is not None:
            stage_callback("Seismic Analysis", total_steps, total_steps)

        return max_disp, static_dx, static_dy, disp_history, time_history
        
    except Exception as exc:
        if stage_callback is not None:
            stage_callback("Seismic Analysis", 0, 0)
        print(
            f"[Error] Seismic analysis exception: sample span={params.span_length:.2f} m, pier_h={params.pier_height:.2f} m, pier_d={params.pier_diameter:.2f} m, record={record.id}, error={exc}"
        )
        return float("nan"), float("nan"), float("nan"), [], []
    finally:
        # Resource cleanup only, no return statement here to avoid swallowing exceptions
        ops.wipe()


def analyze_bridge_with_records(
    params: BridgeParams,
    records: Sequence[AccRecord],
    *,
    sample_id: int,
    save_plots: bool = True,
    plot_dir: Path | None = None,
) -> list[ResultRow]:
    """
    Performs time history analysis for multiple ground motion records.
    """
    results: list[ResultRow] = []
    gm_total = len(records)

    if gm_total == 0:
        return results

    for idx, record in enumerate(records, start=1):
        print(f"Analyzing sample {sample_id}, ground motion {idx}/{gm_total}")
        max_disp, _, _, disp_history, time_history = run_one_earthquake(
            params,
            record,
            stage_index=idx,
            stage_total=gm_total,
        )

        results.append(
            {
                "sample_id": sample_id,
                "gm_id": record.id,
                "span_length_m": params.span_length,
                "pier_height_m": params.pier_height,
                "pier_diameter_m": params.pier_diameter,
                "deck_mass_t_per_m": params.deck_mass,
                "max_disp_mid_m": max_disp,
            }
        )

        # Save the time history plot
        if save_plots and plot_dir:
            sample_dir = plot_dir / f"sample_{sample_id:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            plot_path = sample_dir / f"sample_{sample_id:03d}_gm_{record.id:03d}_disp.png"
            plt.figure(figsize=(10, 6))
            plt.plot(time_history, disp_history, label='Displacement Time History', linewidth=1.0)
            plt.title(f'Sample {sample_id} - Earthquake ground motion {record.id}\nDisplacement time history diagram')
            plt.xlabel('Time (s)')
            plt.ylabel('Displacement (m)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

    return results


# =============================================================================
# 5. Parallel Task Definition
# =============================================================================

def _sample_worker(
    sample_index: int,
    params: BridgeParams,
    *,
    gm_ids: Sequence[int],
    gm_folder: Path,
    save_plots: bool = True,
    plot_dir: Path | None = None,
) -> list[ResultRow]:
    """
    Task executed by each process: runs the full ground motion analysis.
    """
    sample_results: list[ResultRow] = []
    try:
        records = [AccRecord.get_acc(gm_folder, id=gm_id) for gm_id in gm_ids]
        sample_results = analyze_bridge_with_records(
            params,
            records,
            sample_id=sample_index,
            save_plots=save_plots,
            plot_dir=plot_dir,
        )

        # Save the CSV file for each sample
        if plot_dir:
            sample_dir = plot_dir / f"sample_{sample_index:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            csv_path = sample_dir / f"sample_{sample_index:03d}_results.csv"

            fieldnames = [
                "sample_id",
                "gm_id",
                "span_length_m",
                "pier_height_m",
                "pier_diameter_m",
                "deck_mass_t_per_m",
                "max_disp_mid_m",
            ]

            with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in sample_results:
                    formatted_row = {
                        key: (f"{value:.6f}" if isinstance(value, float) else value)
                        for key, value in row.items()
                        if key in fieldnames
                    }
                    writer.writerow(formatted_row)
    except Exception as exc:
        print(f"[Error] Analysis failed for sample #{sample_index}: {exc}")
        for gm_id in gm_ids:
            sample_results.append(
                {
                    "sample_id": sample_index,
                    "gm_id": gm_id,
                    "span_length_m": params.span_length,
                    "pier_height_m": params.pier_height,
                    "pier_diameter_m": params.pier_diameter,
                    "deck_mass_t_per_m": params.deck_mass,
                    "max_disp_mid_m": float("nan"),
                }
            )
    
    # Return moved outside of finally block to prevent exception swallowing
    return sample_results


def run_parametric_study(samples: Sequence[BridgeParams], gm_ids: Sequence[int], gm_folder: Path, save_plots: bool = True, plot_dir: Path | None = None) -> list[ResultRow]:
    """
    Executes the seismic analysis for all samples.
    """
    if not samples:
        return []

    all_results: list[ResultRow] = []