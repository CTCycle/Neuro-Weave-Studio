# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false
from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

###############################################################################
def process_rss_bytes(pid: int) -> int | None:
    """Return best-effort resident memory without importing an optional package."""

    if pid <= 0:
        return None
    if os.name == "nt":
        return _windows_process_rss(pid)
    statm = Path(f"/proc/{pid}/statm")
    try:
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, IndexError, ValueError):
        return None

###############################################################################
def _windows_process_rss(pid: int) -> int | None:
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000 | 0x0400, False, pid)
        if not handle:
            return None

        ###############################################################################
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        result = psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        kernel32.CloseHandle(handle)
        return int(counters.WorkingSetSize) if result else None
    except (AttributeError, OSError):
        return None

###############################################################################
def process_start_options() -> dict[str, Any]:
    """Return platform-safe options that keep a worker in its own process group."""

    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}

###############################################################################
def terminate_process_tree(pid: int, *, force: bool) -> None:
    """Terminate the worker and descendants without relying on framework cleanup."""

    if pid <= 0:
        return
    if os.name == "nt":
        command = ["taskkill", "/PID", str(pid), "/T"]
        if force:
            command.append("/F")
        try:
            subprocess.run(command, capture_output=True, check=False, timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            return
        return
    termination_signal = getattr(signal, "SIGKILL", signal.SIGTERM) if force else signal.SIGTERM
    try:
        os.killpg(os.getpgid(pid), termination_signal)
    except (OSError, ProcessLookupError):
        try:
            os.kill(pid, termination_signal)
        except (OSError, ProcessLookupError):
            return

###############################################################################
def wait_for_process(process: subprocess.Popen[Any], timeout: float) -> bool:
    try:
        process.wait(timeout=max(0.0, timeout))
    except subprocess.TimeoutExpired:
        return False
    return True

###############################################################################
def sleep_poll(seconds: float) -> None:
    time.sleep(max(0.0, seconds))
