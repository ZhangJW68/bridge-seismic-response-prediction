"""Enable ANSI Virtual Terminal processing on Windows 10 consoles.

Based on https://bugs.python.org/issue30075 and MSDN docs.
Call `enable_vt()` once at program startup.
"""
from __future__ import annotations

import os
import ctypes
import msvcrt
from ctypes import wintypes

aTexit_funcs: list[callable] = []  # keep for unit tests

# Win32 constants
STD_OUTPUT_HANDLE = -11
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
ENABLE_WRAP_AT_EOL_OUTPUT = 0x0002

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# Set up error-checking wrappers
LPDWORD = ctypes.POINTER(wintypes.DWORD)

def _check_bool(result, func, args):  # noqa: D401
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
    return args

kernel32.GetConsoleMode.errcheck = _check_bool  # type: ignore[attr-defined]
kernel32.SetConsoleMode.errcheck = _check_bool  # type: ignore[attr-defined]
kernel32.GetConsoleMode.argtypes = (wintypes.HANDLE, LPDWORD)  # type: ignore[attr-defined]
kernel32.SetConsoleMode.argtypes = (wintypes.HANDLE, wintypes.DWORD)  # type: ignore[attr-defined]


def _get_console_handle() -> wintypes.HANDLE:
    """Return handle to CONOUT$ suitable for Get/SetConsoleMode."""
    fd = os.open("CONOUT$", os.O_RDWR)
    try:
        return wintypes.HANDLE(msvcrt.get_osfhandle(fd))  # type: ignore[arg-type]
    finally:
        os.close(fd)


def enable_vt() -> None:
    """Enable VT processing on Windows 10 console if possible."""
    if os.name != "nt":
        return  # non-Windows – nothing to do

    try:
        handle = _get_console_handle()
        mode = wintypes.DWORD()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_WRAP_AT_EOL_OUTPUT
        if new_mode != mode.value:
            kernel32.SetConsoleMode(handle, new_mode)
            # Restore previous mode at exit
            import atexit

            def _restore() -> None:  # noqa: WPS430
                try:
                    kernel32.SetConsoleMode(handle, mode.value)
                except Exception:  # pragma: no cover
                    pass

            atexit.register(_restore)
            aTexit_funcs.append(_restore)
    except Exception:
        # Silently ignore (e.g. old Windows versions, non-interactive shells)
        pass
