"""Compile a CUDA library once across every process asking for it (a file lock, a temporary file renamed into
place)."""
from __future__ import annotations

import fcntl
import os
import subprocess
from pathlib import Path


def compile_locked(lib, command, needed=None, *, prepare=None, run=None):
    """Compile `lib` once across every process asking for it: `command` is the
    compiler's argument list WITHOUT its output (`-o` goes in after the compiler's
    name, naming a temporary file beside `lib` that is renamed into place when
    the compiler exits 0, so no reader ever sees a partial library; the source
    stays the last argument); `needed()` says whether
    a build is due (default: `lib` missing) and is asked again under the lock,
    so a process that waited while another built finds nothing left to do;
    `prepare()` runs under the lock before the compiler (a generated header).
    The lock is `flock` on `<lib>.lock`; a filesystem that refuses it still gets
    the temporary-and-rename, which alone leaves only the double compile."""
    lib = Path(lib)
    needed = needed or (lambda: not lib.exists())
    if not needed():
        return lib
    run = run or subprocess.run
    lib.parent.mkdir(parents=True, exist_ok=True)
    with open(lib.with_name(lib.name + ".lock"), "a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            locked = True
        except OSError:
            locked = False
        try:
            if locked and not needed():
                return lib
            if prepare is not None:
                prepare()
            tmp = lib.with_name(f"{lib.name}.{os.getpid()}.tmp")
            try:
                run([command[0], "-o", str(tmp), *command[1:]], check=True)
                os.replace(tmp, lib)
            finally:
                tmp.unlink(missing_ok=True)
        finally:
            if locked:
                fcntl.flock(fh, fcntl.LOCK_UN)
    return lib
