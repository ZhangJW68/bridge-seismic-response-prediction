# paramters_sampling.py
# -----------------------------------------------------------------------------
#  Cantilever Pier Parameter Definition & Latin Hypercube Sampling (LHS)
#
#  Authors: Jianwei Zhang, Chao Yu, Xiya Liu
#  Affiliation: College of Civil Engineering, Tongji University
#
#  Copyright © 2025 Jianwei Zhang, Chao Yu, Xiya Liu. All rights reserved.
# -----------------------------------------------------------------------------

from __future__ import annotations
import math
import os
import csv
from typing import Annotated, ClassVar

import numpy as np
import numpy.random as npr
from pydantic import BaseModel, Field, computed_field, ConfigDict

import openseespy.opensees as ops

# =============================================================================
# 1. Cantilever Pier Parameter Model (Pydantic)
# =============================================================================

class PierParams(BaseModel):
    """
    Pydantic model for cantilever pier parameters.

    Core input parameters:
        h      : Pier height (m)
        d      : Circular pier diameter (m)
        m_top  : Top concentrated mass (t, corresponding to kN·s²/m in OpenSees)
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(validate_assignment=True)

    h: Annotated[
        float,
        Field(ge=5.0, le=30.0, description="Pier height (m)")
    ]
    d: Annotated[
        float,
        Field(ge=1.0, le=4.0, description="Circular pier diameter (m)")
    ]
    m_top: Annotated[
        float,
        Field(ge=50.0, le=2000.0, description="Superstructure concentrated mass (t)")
    ]

    # Convenient cross-section properties (automatically computed from inputs)

    @computed_field(description="Cross-sectional area A (m²)")
    @property
    def A(self) -> float:
        return math.pi * self.d ** 2 / 4.0

    @computed_field(description="Moment of inertia Iz (m⁴)")
    @property
    def Iz(self) -> float:
        return math.pi * self.d ** 4 / 64.0


# =============================================================================
# 2. Latin Hypercube Sampling (LHS)
# =============================================================================

class LHSSampler:
    """
    Latin Hypercube Sampling utility (uniform distribution version only, simple and effective).
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


def generate_pier_samples(
    n_samples: int,
    h_bounds: tuple[float, float] = (8.0, 20.0),
    d_bounds: tuple[float, float] = (1.5, 3.0),
    m_bounds: tuple[float, float] = (200.0, 1000.0),
    seed: int | None = 42,
) -> list[PierParams]:
    """
    Generates a list of pier parameter samples using LHS.
    """
    base_seed = seed

    h_samples = LHSSampler.lhs_uniform(
        n_samples, h_bounds, seed=base_seed
    )
    d_samples = LHSSampler.lhs_uniform(
        n_samples, d_bounds, seed=None if base_seed is None else base_seed + 1
    )
    m_samples = LHSSampler.lhs_uniform(
        n_samples, m_bounds, seed=None if base_seed is None else base_seed + 2
    )

    samples: list[PierParams] = []
    for i in range(n_samples):
        params = PierParams(
            h=float(h_samples[i]),
            d=float(d_samples[i]),
            m_top=float(m_samples[i]),
        )
        samples.append(params)

    return samples