from __future__ import annotations

"""
多进程进度条工具
----------------
将 Rich 进度条与多进程队列通信封装为通用工具，供主模块直接导入使用。
"""

from typing import Any, Callable, Iterable, Sequence
import multiprocessing
import time
from multiprocessing import Process, Queue

from loguru import logger
from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    SpinnerColumn,
)

import os

# 在 Windows 控制台启用 ANSI VT 模式，避免滚轮滚动导致 Rich 输出失效
if os.name == "nt":
    try:
        from utils.win_vt import enable_vt as _enable_vt  # noqa: E402
        _enable_vt()
    except Exception:
        pass


# 刷新频率与主循环更新时间间隔
REFRESH_RATE = 20
UPDATE_INTERVAL = 0.05

# 子进程心跳超时（秒）。若超过该时间未收到任何进度上报，则判定为超时并终止任务
TIMEOUT_INTERVAL_SECONDS = 60.0


ProgressMessage = dict[str, int | str]


def send_progress_message(queue: Queue[ProgressMessage], worker: str, stage: str, step: int, total: int) -> None:
    """发送进度消息到队列"""
    queue.put({"worker": worker, "stage": stage, "step": step, "total": total})


def create_overall_progress() -> Progress:
    """创建总体进度条"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed}/{task.total} 完成)", justify="right"),
        TimeElapsedColumn(),
    )


def create_worker_state(name: str, stages: list[tuple[str, int]]) -> dict[str, Any]:
    """为每个工作进程创建状态信息"""
    stage_progress = Progress(
        TextColumn("[bold cyan]{task.fields[stage_name]}", justify="right"),
        BarColumn(),
        TextColumn("({task.completed}/{task.total})", justify="right"),
        TimeElapsedColumn(),
    )
    overall_progress_worker = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("({task.completed}/{task.total} 阶段)", justify="right"),
        TimeElapsedColumn(),
    )
    overall_task_worker_id = overall_progress_worker.add_task("阶段完成进度", total=len(stages))

    return {
        "stage_progress": stage_progress,
        "overall_progress": overall_progress_worker,
        "overall_task_id": overall_task_worker_id,
        "current_stage_name": None,
        "current_stage_task_id": None,
        "stages_done": 0,
        "stages_total": len(stages),
    }


def handle_stage_change(state: dict[str, Any], stage: str, total: int) -> None:
    """处理阶段变更"""
    current_task_id = state["current_stage_task_id"]
    if current_task_id is not None:
        state["stage_progress"].update(current_task_id, visible=False)
        state["stage_progress"].stop_task(current_task_id)

    new_task_id = state["stage_progress"].add_task(
        description=f"正在进行 {stage}",
        total=total,
        stage_name=stage,
    )
    state["current_stage_name"] = stage
    state["current_stage_task_id"] = new_task_id


def handle_stage_completion(state: dict[str, Any]) -> None:
    """处理阶段完成"""
    current_task_id = state["current_stage_task_id"]
    if current_task_id is not None:
        state["stage_progress"].update(current_task_id, visible=False)
        state["stage_progress"].stop_task(current_task_id)
        state["stages_done"] += 1
        state["overall_progress"].update(state["overall_task_id"], advance=1)
        state["current_stage_name"] = None
        state["current_stage_task_id"] = None


def process_queue_messages(
    queue: Queue[ProgressMessage],
    worker_state: dict[str, dict[str, Any]],
    overall_progress: Progress,
    overall_task_id: Any,
    finished_workers: int,
    last_report_time: dict[str, float],
) -> int:
    """处理队列中的所有消息"""
    while not queue.empty():
        message = queue.get()
        worker = message["worker"]
        stage = message["stage"]
        step = message["step"]
        total = message["total"]
        state = worker_state[worker]

        # 刷新该 worker 的最后上报时间
        last_report_time[worker] = time.time()

        if stage == "complete":
            if state.get("finished") is not True:
                current_task_id = state["current_stage_task_id"]
                if current_task_id is not None:
                    state["stage_progress"].update(current_task_id, visible=False)
                    state["stage_progress"].stop_task(current_task_id)
                    state["current_stage_task_id"] = None
                    state["current_stage_name"] = None
                try:
                    state["overall_progress"].stop_task(state["overall_task_id"])
                except Exception:
                    pass

                overall_progress.update(overall_task_id, advance=1)
                finished_workers += 1
                state["finished"] = True
            continue

        if state["current_stage_name"] != stage:
            handle_stage_change(state, stage, total)

        current_task_id = state["current_stage_task_id"]
        if current_task_id is not None:
            state["stage_progress"].update(current_task_id, completed=step, total=total)
            if step >= total:
                handle_stage_completion(state)

    return finished_workers


def update_display(
    layout: Layout,
    worker_names: list[str],
    worker_state: dict[str, dict[str, Any]],
    overall_progress: Progress,
) -> None:
    """更新界面显示"""
    worker_panels = []
    for name in worker_names:
        if name not in worker_state:
            continue
        st = worker_state[name]
        if st.get("finished"):
            continue
        panel_group = Group(st["overall_progress"], st["stage_progress"])
        panel = Panel(panel_group, title=name, border_style="cyan")
        worker_panels.append(panel)

    layout["left"].update(Group(*worker_panels))
    right_panel = Panel(Group(overall_progress), title="总体进度", border_style="magenta")
    layout["right"].update(right_panel)


def worker_task_wrapper(worker_name: str, func: Any, args: tuple[Any, ...], queue: Queue[ProgressMessage]) -> None:
    """工作进程任务包装器"""
    try:
        func(*args, progress_queue=queue, worker_name=worker_name)
    except Exception as e:
        logger.error(f"Worker {worker_name} failed: {e}")
        send_progress_message(queue, worker_name, "complete", 0, 0)
        raise


StageFactory = Callable[[tuple[Any, ...], int], list[tuple[str, int]]]
TaskNameFactory = Callable[[tuple[Any, ...], int], str]


def progress_map(
    func: Any,
    iterable: Iterable[Any],
    *,
    description: str = "Processing",
    max_concurrency: int = 4,
    stage_factory: StageFactory | Sequence[tuple[str, int]] | None = None,
    task_name_getter: TaskNameFactory | None = None,
) -> None:
    """使用 Rich 进度条显示并行任务执行情况。

    func 需要支持关键字参数 progress_queue 与 worker_name。
    """

    raw_tasks = list(iterable)
    if not raw_tasks:
        return

    tasks: list[tuple[Any, ...]] = [
        task if isinstance(task, tuple) else (task,) for task in raw_tasks
    ]

    def _default_stage_factory(_: tuple[Any, ...], __: int) -> list[tuple[str, int]]:
        return [("任务执行", 1)]

    resolved_stage_factory: StageFactory
    if isinstance(stage_factory, Sequence) and not isinstance(stage_factory, (str, bytes)):
        default_stage_def = [
            (str(stage_name), int(stage_total))
            for stage_name, stage_total in stage_factory  # type: ignore[misc]
        ]

        def resolved_stage_factory(_: tuple[Any, ...], __: int) -> list[tuple[str, int]]:
            return list(default_stage_def)
    else:
        resolved_stage_factory = stage_factory or _default_stage_factory

    def _default_name_getter(_: tuple[Any, ...], idx: int) -> str:
        return f"{description}_{idx + 1:02d}"

    name_getter = task_name_getter or _default_name_getter

    stages_per_task: list[list[tuple[str, int]]] = [
        list(resolved_stage_factory(task, i)) for i, task in enumerate(tasks)
    ]

    task_names = [name_getter(task, i) for i, task in enumerate(tasks)]

    queue: Queue[ProgressMessage] = multiprocessing.Queue()
    active_processes: dict[str, Process] = {}
    worker_state: dict[str, dict[str, Any]] = {}
    last_report_time: dict[str, float] = {}

    task_index = 0

    def launch_next() -> None:
        nonlocal task_index
        if task_index >= len(tasks):
            return
        task_name = task_names[task_index]
        task_args = tasks[task_index]
        p = Process(target=worker_task_wrapper, args=(task_name, func, task_args, queue))
        p.start()
        active_processes[task_name] = p
        worker_state[task_name] = create_worker_state(task_name, stages_per_task[task_index])
        # 记录启动时的初始上报时间（避免未及时上报导致被立即误判）
        last_report_time[task_name] = time.time()
        task_index += 1

    for _ in range(min(max_concurrency, len(tasks))):
        launch_next()

    console = Console(force_terminal=True)
    overall_progress = create_overall_progress()
    overall_task_id = overall_progress.add_task(f"总进度 - {description}", total=len(tasks))

    layout = Layout()
    layout.split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=1))
    finished_workers = 0

    with Live(layout, refresh_per_second=REFRESH_RATE, console=console, screen=False, transient=True):
        while finished_workers < len(tasks):
            while len(active_processes) < max_concurrency and task_index < len(tasks):
                launch_next()

            finished_workers = process_queue_messages(
                queue,
                worker_state,
                overall_progress,
                overall_task_id,
                finished_workers,
                last_report_time,
            )

            # 超时监控：若超过 TIMEOUT_INTERVAL_SECONDS 未收到上报，则终止并标记完成
            now_ts = time.time()
            for name, proc in list(active_processes.items()):
                last_ts = last_report_time.get(name, 0.0)
                state = worker_state.get(name)
                is_finished = bool(state and state.get("finished"))
                if not is_finished and last_ts > 0 and (now_ts - last_ts) > TIMEOUT_INTERVAL_SECONDS:
                    try:
                        logger.warning(f"Worker {name} 超时：{now_ts - last_ts:.1f}s 未收到上报，将终止该任务")
                    except Exception:
                        pass
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    try:
                        proc.join()
                    except Exception:
                        pass

                    # 标记该 worker 完成并更新总体进度
                    if state is not None and state.get("finished") is not True:
                        current_task_id = state.get("current_stage_task_id")
                        if current_task_id is not None:
                            try:
                                state["stage_progress"].update(current_task_id, visible=False)
                                state["stage_progress"].stop_task(current_task_id)
                            except Exception:
                                pass
                        state["current_stage_task_id"] = None
                        state["current_stage_name"] = None
                        try:
                            state["overall_progress"].stop_task(state["overall_task_id"])
                        except Exception:
                            pass
                        overall_progress.update(overall_task_id, advance=1)
                        finished_workers += 1
                        state["finished"] = True

                    # 从活动进程中移除
                    try:
                        del active_processes[name]
                    except Exception:
                        pass

            # 移除已自然结束的进程
            for name, proc in list(active_processes.items()):
                if not proc.is_alive():
                    proc.join()
                    del active_processes[name]

            update_display(layout, task_names, worker_state, overall_progress)
            time.sleep(UPDATE_INTERVAL)

    for proc in active_processes.values():
        proc.join()
    console.print(f"[bold green]所有{description}任务已完成！")


