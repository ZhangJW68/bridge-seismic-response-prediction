from pathlib import Path
from collections.abc import Mapping, Sequence
from functools import cached_property, lru_cache
from typing import Callable, ClassVar, Literal, TextIO, cast, override

import numpy as np

GRAVITY = 9.80665
CM_PER_M = 100.0
ACCUnit = Literal["g", "gal", "cm/s^2", "cm/s2", "m/s^2", "m/s2"]
ACC_UNIT_FACTORS: dict[str, float] = {
    "g": GRAVITY,
    "gal": 0.01,
    "cm/s^2": 0.01,
    "cm/s2": 0.01,
    "m/s^2": 1.0,
    "m/s2": 1.0,
}


class AccRecord:
    IM_FIELD_SPECS: ClassVar[tuple[tuple[str, str, str], ...]] = (
        ("PGA", "PGA", "g"),
        ("PGV", "PGV", "cm/s"),
        ("PGD", "PGD", "cm"),
        ("VAratio", "VAratio", "s"),
        ("Ia", "Ia", "m/s"),
        ("SD5_95", "SD5_95", "s"),
        ("SD5_75", "SD5_75", "s"),
        ("Arms", "Arms", "g"),
        ("Vrms", "Vrms", "cm/s"),
        ("Drms", "Drms", "cm"),
        ("Ic", "Ic", "g^1.5·s^0.5"),
        ("CAV", "CAV", "cm/s"),
        ("CAV5", "CAV5", "cm/s"),
        ("CAD", "CAD", "cm"),
        ("ASI", "ASI", "g·s"),
        ("VSI", "VSI", "cm"),
        ("Sa02", "Sa02", "g"),
        ("Sv02", "Sv02", "cm/s"),
        ("Sd02", "Sd02", "cm"),
        ("Sa10", "Sa10", "g"),
        ("Sv10", "Sv10", "cm/s"),
        ("Sd10", "Sd10", "cm"),
        ("Sa20", "Sa20", "g"),
        ("Sv20", "Sv20", "cm/s"),
        ("Sd20", "Sd20", "cm"),
    )

    @staticmethod
    def _resolve_acc_unit(unit: ACCUnit) -> float:
        if not unit:
            raise ValueError("acc_unit 不能为空")
        key = unit.strip().lower().replace(" ", "")
        key = key.replace("m/s2", "m/s^2").replace("cm/s2", "cm/s^2")
        factor = ACC_UNIT_FACTORS.get(key)
        if factor is None:
            raise ValueError(f"不支持的加速度单位: {unit!r}")
        return factor

    def __init__(
        self,
        id: int | None = None,
        folder: Path | str | None = None,
        *,
        acc: Sequence[float] | np.ndarray | None = None,
        time: Sequence[float] | np.ndarray | None = None,
        dt: float | None = None,
        metadata: Mapping[str, object] | None = None,
        acc_unit: ACCUnit = "g",
    ):
        self.id: int = int(id) if id is not None else -1
        self.folder: Path | None = Path(folder) if folder is not None else None

        self.acc_unit: str = acc_unit
        self._acc_scale: float = self._resolve_acc_unit(acc_unit)

        self._acc_override: tuple[float, ...] | None = None
        if acc is not None:
            scale = self._acc_scale
            self._acc_override = tuple(float(value) * scale for value in acc)

        self._time_override: tuple[float, ...] | None = None
        if time is not None:
            self._time_override = tuple(float(value) for value in time)

        if (
            self._acc_override is not None
            and self._time_override is not None
            and len(self._acc_override) != len(self._time_override)
        ):
            raise ValueError("time 序列与 acc 序列的长度必须一致")

        self._dt_override: float | None = float(dt) if dt is not None else None
        if self.folder is None and self._acc_override is None:
            raise ValueError("必须提供 folder 或 acc 序列以构建 AccRecord")

        if (
            self._acc_override is not None
            and self.folder is None
            and self._time_override is None
            and self._dt_override is None
        ):
            raise ValueError("直接传入 acc 序列时，必须提供 time 序列或 dt")

        if (
            self._time_override is not None
            and self._dt_override is not None
            and len(self._time_override) >= 2
        ):
            inferred_dt = self._compute_dt_from_time()
            dt_override_value = float(self._dt_override)
            rounded_dt = float(round(dt_override_value, 3))
            if rounded_dt <= 0.0:
                raise ValueError("dt 必须为正数")
            if abs(inferred_dt - rounded_dt) > 1e-6:
                raise ValueError("time 序列与 dt 不一致")
            self._dt_override = rounded_dt

        self.metadata: dict[str, object] = dict(metadata) if metadata is not None else {}

    @cached_property
    def acc_series(self) -> list[float]:
        if self._acc_override is not None:
            return list(self._acc_override)
        return self.load_acc()

    @cached_property
    def vel_series(self) -> list[float]:
        return self.acc_series_to_vel_series(self.acc_series)

    @cached_property
    def time_series(self) -> list[float]:
        if self._time_override is not None:
            return list(self._time_override)
        return [i * self.dt for i in range(self.n_steps)]

    @cached_property
    def disp_series(self) -> list[float]:
        return self.vel_series_to_disp_series(self.vel_series)

    @cached_property
    def _acc_array(self) -> np.ndarray:
        return np.asarray(self.acc_series, dtype=float)

    @cached_property
    def _vel_array(self) -> np.ndarray:
        if not self._acc_array.size:
            return np.asarray([], dtype=float)
        return np.cumsum(self._acc_array, dtype=float) * float(self.dt)

    @cached_property
    def _disp_array(self) -> np.ndarray:
        if not self._vel_array.size:
            return np.asarray([], dtype=float)
        return np.cumsum(self._vel_array, dtype=float) * float(self.dt)

    @cached_property
    def _time_array(self) -> np.ndarray:
        size = self._acc_array.size
        if size == 0:
            return np.asarray([], dtype=float)
        return np.arange(size, dtype=float) * float(self.dt)

    @cached_property
    def _pga_ms2(self) -> float:
        if not self._acc_array.size:
            return 0.0
        return float(np.max(np.abs(self._acc_array)))

    @cached_property
    def dt(self) -> float:
        return self.get_dt()

    @cached_property
    def PGA(self) -> float:
        if self._pga_ms2 == 0.0:
            return 0.0
        return self._pga_ms2 / GRAVITY

    @cached_property
    def PGV(self) -> float:
        if not self._vel_array.size:
            return 0.0
        return float(np.max(np.abs(self._vel_array))) * CM_PER_M

    @cached_property
    def PGD(self) -> float:
        if not self._disp_array.size:
            return 0.0
        return float(np.max(np.abs(self._disp_array))) * CM_PER_M

    @cached_property
    def VAratio(self) -> float:
        pga_ms2 = self._pga_ms2
        if pga_ms2 == 0.0 or not self._vel_array.size:
            return 0.0
        return float(np.max(np.abs(self._vel_array))) / pga_ms2

    @cached_property
    def _arias_intensity_cum(self) -> np.ndarray:
        if not self._acc_array.size:
            return np.asarray([], dtype=float)
        scale = np.pi / (2.0 * GRAVITY)
        cumulative = np.cumsum(self._acc_array ** 2, dtype=float) * float(self.dt)
        cumulative *= scale
        return np.maximum.accumulate(cumulative)

    @cached_property
    def Ia(self) -> float:
        if not self._arias_intensity_cum.size:
            return 0.0
        return float(self._arias_intensity_cum[-1])

    def _significant_interval(self, lower_frac: float, upper_frac: float) -> tuple[float, float]:
        if not self._arias_intensity_cum.size:
            return 0.0, 0.0
        total = float(self._arias_intensity_cum[-1])
        if total <= 0.0:
            return 0.0, 0.0
        time = self._time_array
        lower_time = float(np.interp(total * lower_frac, self._arias_intensity_cum, time))
        upper_time = float(np.interp(total * upper_frac, self._arias_intensity_cum, time))
        return lower_time, upper_time

    @cached_property
    def SD5_95(self) -> float:
        t_start, t_end = self._significant_interval(0.05, 0.95)
        return max(0.0, t_end - t_start)

    @cached_property
    def SD5_75(self) -> float:
        t_start, t_end = self._significant_interval(0.05, 0.75)
        return max(0.0, t_end - t_start)

    def _rms_between(self, values: np.ndarray, t_start: float, t_end: float) -> float:
        if not values.size or t_end <= t_start:
            return 0.0
        time = self._time_array
        mask = (time >= t_start) & (time <= t_end)
        if not np.any(mask):
            return 0.0
        selected_values = values[mask]
        selected_time = time[mask]
        if selected_values.size < 2:
            return float(np.abs(selected_values).mean())
        integral = float(np.trapezoid(selected_values ** 2, x=selected_time))
        duration = t_end - t_start
        if duration <= 0.0:
            return 0.0
        return float(np.sqrt(integral / duration))

    @cached_property
    def Arms(self) -> float:
        t_start, t_end = self._significant_interval(0.05, 0.95)
        rms_value = self._rms_between(self._acc_array, t_start, t_end)
        if rms_value == 0.0:
            return 0.0
        return rms_value / GRAVITY

    @cached_property
    def Vrms(self) -> float:
        t_start, t_end = self._significant_interval(0.05, 0.95)
        rms_value = self._rms_between(self._vel_array, t_start, t_end)
        return rms_value * CM_PER_M

    @cached_property
    def Drms(self) -> float:
        t_start, t_end = self._significant_interval(0.05, 0.95)
        rms_value = self._rms_between(self._disp_array, t_start, t_end)
        return rms_value * CM_PER_M

    @cached_property
    def Ic(self) -> float:
        t_start, t_end = self._significant_interval(0.05, 0.95)
        duration = max(0.0, t_end - t_start)
        if duration <= 0.0:
            return 0.0
        arms_value = self.Arms
        if arms_value == 0.0:
            return 0.0
        return float((arms_value ** 1.5) * np.sqrt(duration))

    @cached_property
    def CAV(self) -> float:
        if not self._acc_array.size:
            return 0.0
        integral = float(np.trapezoid(np.abs(self._acc_array), dx=self.dt))
        return integral * CM_PER_M

    @cached_property
    def CAV5(self) -> float:
        if not self._acc_array.size:
            return 0.0
        threshold = 0.05 * GRAVITY
        time = self._time_array
        mask = np.abs(self._acc_array) >= threshold
        if not np.any(mask):
            return 0.0
        integral = float(np.trapezoid(np.abs(self._acc_array[mask]), x=time[mask]))
        return integral * CM_PER_M

    @cached_property
    def CAD(self) -> float:
        if not self._disp_array.size:
            return 0.0
        integral = float(np.trapezoid(np.abs(self._disp_array), dx=self.dt))
        return integral * CM_PER_M

    def _response_values(self, periods: Sequence[float]) -> dict[str, np.ndarray]:
        return self.response_spectrum(periods)

    @cached_property
    def ASI(self) -> float:
        periods = [float(p) for p in np.linspace(0.1, 0.5, 41, dtype=float)]
        spectrum = self._response_values(periods)
        sa_g = spectrum['Sa'] / GRAVITY
        return float(np.trapezoid(sa_g, spectrum['T']))

    @cached_property
    def VSI(self) -> float:
        periods = [float(p) for p in np.linspace(0.1, 2.5, 241, dtype=float)]
        spectrum = self._response_values(periods)
        sv_cm_s = spectrum['Sv'] * CM_PER_M
        return float(np.trapezoid(sv_cm_s, spectrum['T']))

    def _spectral_triplet(self, period: float) -> tuple[float, float, float]:
        spectrum = self._response_values((float(period),))
        sa = float(spectrum['Sa'][0]) / GRAVITY
        sv = float(spectrum['Sv'][0]) * CM_PER_M
        sd = float(spectrum['Sd'][0]) * CM_PER_M
        return sa, sv, sd

    @cached_property
    def Sa02(self) -> float:
        sa, _, _ = self._spectral_triplet(0.2)
        return sa

    @cached_property
    def Sv02(self) -> float:
        _, sv, _ = self._spectral_triplet(0.2)
        return sv

    @cached_property
    def Sd02(self) -> float:
        _, _, sd = self._spectral_triplet(0.2)
        return sd

    @cached_property
    def Sa10(self) -> float:
        sa, _, _ = self._spectral_triplet(1.0)
        return sa

    @cached_property
    def Sv10(self) -> float:
        _, sv, _ = self._spectral_triplet(1.0)
        return sv

    @cached_property
    def Sd10(self) -> float:
        _, _, sd = self._spectral_triplet(1.0)
        return sd

    @cached_property
    def Sa20(self) -> float:
        sa, _, _ = self._spectral_triplet(2.0)
        return sa

    @cached_property
    def Sv20(self) -> float:
        _, sv, _ = self._spectral_triplet(2.0)
        return sv

    @cached_property
    def Sd20(self) -> float:
        _, _, sd = self._spectral_triplet(2.0)
        return sd

    def print_intensity_measures(self, *, precision: int = 4, file: TextIO | None = None) -> None:
        """打印当前实例全部已实现的地震动指标(IM)。"""
        if file is None:
            import sys

            output = sys.stdout
        else:
            output = file

        name_width = max(len(name) for name, _, _ in self.IM_FIELD_SPECS)
        for name, attr, unit in self.IM_FIELD_SPECS:
            value = float(getattr(self, attr))
            formatted = f"{value:.{precision}f}" if np.isfinite(value) else str(value)
            print(f"{name:<{name_width}} : {formatted} {unit}", file=output)

    @cached_property
    def n_steps(self) -> int:
        return self.get_n_steps()

    @classmethod
    def get_one_random_acc(cls, folder: Path | str, max_id: int = 80, seed: int | None = None, acc_unit: ACCUnit = "g") -> 'AccRecord':
        import random
        folder_path = Path(folder)
        # 仅保留文件名（不含扩展名）可转换为整数且不超过给定上限的记录（本例中 max_id=80）
        valid_acc_files = [
            p for p in folder_path.glob('*.acc')
            if p.stem.isdigit() and int(p.stem) <= max_id
        ]

        if not valid_acc_files:
            raise FileNotFoundError(
                f"未找到满足条件 (*.acc 且文件名为整数 ≤ {max_id}) 的地震记录: {folder_path}"
            )

        if seed is not None:
            random.seed(seed)

        acc_file = random.choice(valid_acc_files)
        return cls(int(acc_file.stem), folder_path, acc_unit=acc_unit)

    @classmethod
    def get_acc(cls, folder: Path | str, id: int = 1, acc_unit: ACCUnit = "g") -> 'AccRecord':
        folder_path = Path(folder)
        acc_file = folder_path / f"{id}.acc"
        if not acc_file.exists():
            raise FileNotFoundError(f"加载地震记录失败: {acc_file}")
        return cls(id, folder_path, acc_unit=acc_unit)

    @classmethod
    def from_series(
        cls,
        acc: Sequence[float] | np.ndarray,
        *,
        time: Sequence[float] | np.ndarray | None = None,
        dt: float | None = None,
        id: int | None = None,
        metadata: Mapping[str, object] | None = None,
        acc_unit: ACCUnit = "g",
    ) -> 'AccRecord':
        """通过直接传入加速度（以及可选的时间、dt、acc_unit）构造 `AccRecord`。"""
        if len(acc) == 0:
            raise ValueError("acc 序列不能为空")
        return cls(id=id, folder=None, acc=acc, time=time, dt=dt, metadata=metadata, acc_unit=acc_unit)

    def load_acc(self, suffix: str = 'acc') -> list[float]:
        if self._acc_override is not None:
            return list(self._acc_override)

        if self.folder is None:
            raise ValueError("当前 AccRecord 未绑定地震记录文件夹，无法加载地震记录")

        acc_file = self.folder / f"{self.id}.{suffix}"
        if not acc_file.exists():
            raise FileNotFoundError(f"加载地震记录失败: {acc_file}")

        with open(acc_file, 'r') as f:
            lines = f.readlines()
            scale = self._acc_scale
            acc_series = [float(line.split()[0]) * scale for line in lines]
        return acc_series

    def _compute_dt_from_time(self) -> float:
        if self._time_override is None or len(self._time_override) < 2:
            raise ValueError("无法从长度不足 2 的时间序列计算 dt，请显式提供 dt")

        time_array = np.asarray(self._time_override, dtype=float)
        diffs = np.diff(time_array)
        if diffs.size == 0:
            raise ValueError("时间序列差分长度为 0，无法计算 dt")

        rounded_diffs = np.round(diffs, 8)
        unique_diffs_array = np.unique(rounded_diffs).astype(float, copy=False)
        if unique_diffs_array.size == 0:
            raise ValueError("时间序列差分为空，无法计算 dt")

        unique_diffs = cast(list[float], unique_diffs_array.tolist())
        min_diff = min(unique_diffs)
        max_diff = max(unique_diffs)
        if max_diff - min_diff < 1e-3:
            mean_diff = float(sum(unique_diffs) / len(unique_diffs))
            dt_candidate = float(round(mean_diff, 3))
            if dt_candidate <= 0.0:
                raise ValueError("推断得到的 dt 非正")
            return dt_candidate
        print("最大时间间隔误差", max_diff - min_diff)
        mean_diff = float(sum(unique_diffs) / len(unique_diffs))
        dt_candidate = float(round(mean_diff, 3))
        if dt_candidate <= 0.0:
            raise ValueError("推断得到的 dt 非正")
        return dt_candidate

    def get_dt(self, id: int | None = None, dt_file: Path | str = 'dt.txt') -> float:
        if self._dt_override is not None:
            return self._dt_override

        if self._time_override is not None:
            dt_value = self._compute_dt_from_time()
            self._dt_override = dt_value
            return dt_value

        if self.folder is None:
            raise ValueError("当前 AccRecord 未绑定地震记录文件夹，无法读取 dt")

        target_id = int(self.id if id is None else id)
        dt_path = self.folder / Path(dt_file)
        if not dt_path.exists():
            raise FileNotFoundError(f"加载地震时间步长失败: {dt_path}")

        with open(dt_path, 'r') as f:
            lines = f.readlines()
            dt_value = float(lines[target_id - 1])

        self._dt_override = dt_value
        return dt_value

    def get_n_steps(self, id: int | None = None, n_steps_file: Path | str = 'motionStep.txt') -> int:
        if self._time_override is not None:
            return len(self._time_override)
        if self._acc_override is not None:
            return len(self._acc_override)

        if self.folder is None:
            raise ValueError("当前 AccRecord 未绑定地震记录文件夹，无法读取步数信息")

        target_id = int(self.id if id is None else id)
        n_steps_path = self.folder / Path(n_steps_file)
        if not n_steps_path.exists():
            raise FileNotFoundError(f"加载地震步数失败: {n_steps_path}")

        with open(n_steps_path, 'r') as f:
            lines = f.readlines()
            n_steps = int(lines[target_id - 1])
            return n_steps

    def acc_series_to_vel_series(self, acc_series: list[float]) -> list[float]:
        """将加速度序列积分为速度序列（列表，单位与输入一致）。

        纯 Python 实现，避免依赖 Numpy，保证返回的元素全部为内置 ``float`` 类型。
        ``vel[0]`` 取 0.0，对应初始时刻速度。
        """
        vel: list[float] = [0.0]
        cumulative: float = 0.0
        dt = self.dt
        for a in acc_series:
            cumulative += a
            vel.append(cumulative * dt)
        return vel

    def vel_series_to_disp_series(self, vel_series: list[float]) -> list[float]:
        """将速度序列积分为位移序列（列表，单位与输入一致）。

        纯 Python 实现，避免依赖 Numpy，保证返回的元素全部为内置 ``float`` 类型。
        ``disp[0]`` 取 0.0，对应初始位移。
        """
        disp: list[float] = [0.0]
        cumulative: float = 0.0
        dt = self.dt
        for v in vel_series:
            cumulative += v
            disp.append(cumulative * dt)
        return disp

    def acc_factor(self, target_PGA: float) -> float:
        return target_PGA / self.PGA

    def interp_acc_series(self, time: list[float] | np.ndarray) -> np.ndarray:
        return np.interp(time, self.time_series, self.acc_series)

    def interp_vel_series(self, time: list[float] | np.ndarray) -> np.ndarray:
        return np.interp(time, self.time_series, self.vel_series)
    
    def interp_disp_series(self, time: list[float] | np.ndarray) -> np.ndarray:
        return np.interp(time, self.time_series, self.disp_series)

    @staticmethod
    def _normalize_periods(periods: Sequence[float] | np.ndarray) -> tuple[float, ...]:
        try:
            normalized = tuple(float(p) for p in periods)
        except TypeError as exc:
            raise TypeError("periods 参数必须是可迭代的数值序列") from exc

        if not normalized:
            raise ValueError("periods 不能为空")

        if any(p < 0.0 for p in normalized):
            raise ValueError("periods 中的周期必须全部为非负数")

        return normalized

    def response_spectrum(self, periods: Sequence[float] | np.ndarray, damping: float = 0.05) -> dict[str, np.ndarray]:
        """计算给定阻尼比下的反应谱。

        参数
        ----
        periods:
            周期列表/数组（单位：秒）。
        damping:
            阻尼比，默认 0.05（5%）。

        返回
        ----
        dict[str, np.ndarray]
            包含 `T`（周期）、`Sd`（最大相对位移）、`Sv`（最大相对速度）、
            `Sa`（最大绝对加速度）、`PSv` 和 `PSa`（伪速度/加速度）的数组。
        """
        periods_key = self._normalize_periods(periods)
        damping_value = float(damping)

        (periods_arr,
         sd,
         sv,
         sa,
         psv,
         psa) = self._response_spectrum(periods_key, damping_value)

        return {
            'T': periods_arr.copy(),
            'Sd': sd.copy(),
            'Sv': sv.copy(),
            'Sa': sa.copy(),
            'PSv': psv.copy(),
            'PSa': psa.copy(),
        }

    @lru_cache(maxsize=None)
    def _response_spectrum(
        self,
        periods_key: tuple[float, ...],
        damping: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not 0.0 <= damping < 1.0:
            raise ValueError("damping 必须位于 [0.0, 1.0) 区间内")

        acc = np.asarray(self.acc_series, dtype=float)
        dt = float(self.dt)
        beta = 0.25
        gamma = 0.5

        periods = np.array(periods_key, dtype=float)
        sd = np.zeros_like(periods, dtype=float)
        sv = np.zeros_like(periods, dtype=float)
        sa = np.zeros_like(periods, dtype=float)
        psv = np.zeros_like(periods, dtype=float)
        psa = np.zeros_like(periods, dtype=float)

        max_ag = np.max(np.abs(acc)) if acc.size else 0.0

        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        a6 = dt * (1.0 - gamma)
        a7 = gamma * dt

        period_values = cast(Sequence[float], periods_key)
        acc_values = cast(Sequence[float], np.asarray(acc, dtype=float).tolist())

        for idx, period_raw in enumerate(period_values):
            period = float(period_raw)
            if period <= 0.0:
                sd[idx] = 0.0
                sv[idx] = 0.0
                sa[idx] = max_ag
                psv[idx] = 0.0
                psa[idx] = max_ag
                continue

            omega = 2.0 * np.pi / period
            k = omega * omega
            c = 2.0 * damping * omega
            k_hat = k + a0 + a1 * c

            u = 0.0
            u_dot = 0.0
            u_ddot = 0.0
            max_u = 0.0
            max_u_dot = 0.0
            max_abs_acc = 0.0

            for ag_raw in acc_values:
                ag = float(ag_raw)
                effective_force = -ag
                p_hat = (
                    effective_force
                    + a0 * u
                    + a2 * u_dot
                    + a3 * u_ddot
                    + c * (a1 * u + a4 * u_dot + a5 * u_ddot)
                )

                u_new = p_hat / k_hat
                u_ddot_new = a0 * (u_new - u) - a2 * u_dot - a3 * u_ddot
                u_dot_new = u_dot + a6 * u_ddot + a7 * u_ddot_new
                abs_acc = u_ddot_new + ag

                max_u = max(max_u, abs(u_new))
                max_u_dot = max(max_u_dot, abs(u_dot_new))
                max_abs_acc = max(max_abs_acc, abs(abs_acc))

                u = u_new
                u_dot = u_dot_new
                u_ddot = u_ddot_new

            sd[idx] = max_u
            sv[idx] = max_u_dot
            sa[idx] = max_abs_acc
            psv[idx] = omega * max_u
            psa[idx] = omega * omega * max_u

        return periods, sd, sv, sa, psv, psa

    @override
    def __str__(self) -> str:
        return f"AccRecord(id={self.id}, PGA={self.PGA:.3f} g, dt={self.dt:.3f} s, n_steps={self.n_steps})"

if __name__ == "__main__":
    acc_folder = Path('./waves')
    record = AccRecord.get_acc(acc_folder, id=2)
    record.print_intensity_measures()
    # print(record)
    import numpy as np
    RS = record.response_spectrum(periods = np.arange(0.0, 10.0, 0.01))
    # print(RS)
    import matplotlib.pyplot as plt
    plot_func = cast(
        Callable[[Sequence[float], Sequence[float]], list[object]],
        plt.plot,
    )
    periods_for_plot = cast(list[float], RS['T'].tolist())
    sa_for_plot = cast(list[float], RS['Sa'].tolist())
    _ = plot_func(periods_for_plot, sa_for_plot)
    show_func = cast(Callable[[], None], plt.show)
    show_func()