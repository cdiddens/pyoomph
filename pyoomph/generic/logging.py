#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  @author Maxim de Wildt <m.dewildt@utwente.nl>
#
#  @section LICENSE
#
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC
#  Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  The main author may be contacted at c.diddens@utwente.nl
#
# ========================================================================

"""Console handling for pyoomph's Python-side output.

Two jobs, both done by the same stdout/stderr wrapper:

* mirroring everything into the problem log file (the original purpose), and
* the MPI console. Under ``mpirun`` every rank writes to the same terminal, so the raw output is
  duplicated once per rank and, because neither Python nor C++ writes a line atomically, chopped
  into interleaved fragments. :py:func:`setup_mpi_console` filters it; the identical policy is
  applied to oomph-lib's C++ output by ``src/logging.cpp``, so both halves of the output agree.

MPI modes (``--mpi-output``):

``condensed``
    Default. Only rank 0 reaches the terminal. The other ranks print the very same lines
    (residuals, dof counts and timings are globally reduced before oomph-lib reports them), so
    nothing is lost - and their stderr still gets through, tagged, so a failure on rank 3 is
    still visible.
``all``
    Every rank, but each line written in one piece and tagged ``[rank N]``. For when the ranks
    really do disagree and that is what is being debugged.
``off``
    No filtering at all.
"""

import sys
import os
from typing import Optional, TextIO, Any

from .. import _pyoomph_core as _pyoomph

_MODES = ("condensed", "all", "off")

# Filled in by setup_mpi_console(); mirrored on the C++ side for oomph-lib's own output.
_mpi_rank = 0
_mpi_nproc = 1
_mpi_mode = "off"


def mpi_console_mode() -> str:
    """The active MPI output mode, one of ``"condensed"``, ``"all"`` or ``"off"``."""
    return _mpi_mode


def mpi_console_is_filtering() -> bool:
    """True if output is being filtered, i.e. more than one rank and a mode other than ``off``."""
    return _mpi_nproc > 1 and _mpi_mode != "off"


class _ConsoleWrapper(object):
    """Line-buffered stand-in for ``sys.stdout``/``sys.stderr``.

    Line buffering is not an optimisation here: it is what makes a line atomic. ``print()``
    reaches ``write()`` in several calls (the arguments, the separators, the newline), and under
    mpirun another rank's fragment lands between them. Holding the characters until the newline
    and writing the line in a single ``write`` removes that, and is also what allows a whole-line
    decision on whether this rank is allowed to print at all.
    """

    def __init__(self, terminal: TextIO, is_stderr: bool = False):
        self.terminal = terminal
        self.is_stderr = is_stderr
        self._pending = ""
        # False while a partial line pushed out by flush() is still open, so that the rank tag is
        # not repeated in the middle of it (print(".", end="", flush=True) in a loop).
        self._at_line_start = True

    # -- policy ---------------------------------------------------------------------------
    def _muted(self) -> bool:
        if not mpi_console_is_filtering():
            return False
        if _mpi_mode == "condensed":
            # stderr always gets through: a traceback on rank 3 is exactly the case where the
            # ranks differ, and swallowing it would leave the run looking like a silent hang.
            return _mpi_rank != 0 and not self.is_stderr
        return False

    def _decorate(self, line: str) -> str:
        if not mpi_console_is_filtering():
            return line
        # stderr is tagged on every rank, rank 0 included: when several ranks raise, their
        # tracebacks arrive interleaved (line-atomic, but mixed), and an untagged one among tagged
        # ones cannot be followed. stdout stays untagged in condensed mode - only rank 0 prints it.
        if _mpi_mode == "all" or self.is_stderr:
            return "[rank " + str(_mpi_rank) + "] " + line
        return line

    # -- file-like interface --------------------------------------------------------------
    def write(self, message: str) -> int:
        if not mpi_console_is_filtering():
            # Serial runs keep the original behaviour exactly, straight through to the terminal.
            self.terminal.write(message)
            _pyoomph._write_to_log_file(message)
            return len(message)
        self._pending += message
        while True:
            nl = self._pending.find("\n")
            if nl < 0:
                break
            line, self._pending = self._pending[:nl], self._pending[nl + 1:]
            self._emit(line)
        return len(message)

    def _emit(self, line: str) -> None:
        out = (self._decorate(line) if self._at_line_start else line) + "\n"
        self._at_line_start = True
        # The log file records what this rank produced regardless of whether it was shown; where no
        # log file is open (every rank but 0) this is a no-op.
        _pyoomph._write_to_log_file(out)
        if self._muted():
            return
        # oomph-lib's C++ output reaches the same terminal through a different buffer, so both
        # sides have to flush per line or the two arrive out of order.
        _pyoomph._flush_console()
        self.terminal.write(out)
        self.terminal.flush()

    def flush(self) -> None:
        # A tagged partial line cannot stay atomic - another rank's fragment lands inside it before
        # the newline arrives - so where a tag is being added, hold it back until the line is
        # complete. Untagged output (rank 0 when condensed) is pushed out, so a progress line built
        # from print(".", end="", flush=True) still appears as it is produced.
        if self._pending and self._decorate("") == "":
            out = self._decorate(self._pending) if self._at_line_start else self._pending
            self._pending = ""
            self._at_line_start = False
            _pyoomph._write_to_log_file(out)
            if not self._muted():
                _pyoomph._flush_console()
                self.terminal.write(out)
        try:
            self.terminal.flush()
        except (ValueError, OSError):
            pass  # interpreter shutdown closed it already

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self) -> int:
        return self.terminal.fileno()

    @property
    def encoding(self) -> Any:
        return getattr(self.terminal, "encoding", "utf-8")

    def writable(self) -> bool:
        return True

    def __del__(self):
        if self.is_stderr:
            sys.stderr = self.terminal
        else:
            sys.stdout = self.terminal


def _install_wrappers() -> None:
    if not isinstance(sys.stdout, _ConsoleWrapper):
        sys.stdout = _ConsoleWrapper(sys.stdout)
    if not isinstance(sys.stderr, _ConsoleWrapper):
        sys.stderr = _ConsoleWrapper(sys.stderr, is_stderr=True)


def pyoomph_activate_logging_to_file() -> None:
    """Start mirroring console output into the problem log file opened on the C++ side."""
    _install_wrappers()


def resolve_mpi_output_mode(explicit: Optional[str] = None) -> str:
    """Pick the MPI output mode from an explicit argument, the environment or the command line.

    Called before the problem's ``argparse`` exists - the MPI banner is printed at import time -
    so ``--mpi-output`` is looked for in ``sys.argv`` by hand here. ``Problem.setup_cmd_line()``
    registers the same flag so that it is documented in ``--help`` and is not passed on to PETSc,
    which reports every unread dash-argument as a possible typo.
    """
    if explicit is not None:
        mode = explicit.strip().lower()
        if mode not in _MODES:
            raise ValueError("Unknown MPI output mode '" + mode + "', expected one of " + ", ".join(_MODES))
        return mode
    mode = os.environ.get("PYOOMPH_MPI_OUTPUT", "")
    for i, a in enumerate(sys.argv[1:]):
        if a.startswith("--mpi-output="):
            mode = a.split("=", 1)[1]
        elif a == "--mpi-output" and i + 2 < len(sys.argv):
            mode = sys.argv[i + 2]
    mode = mode.strip().lower()
    # Lenient on purpose: this runs at import time, long before the problem's argparse exists, and
    # a typo here must not turn into a traceback out of "from pyoomph import *". argparse rejects
    # the value properly (with a usage message) once parse_cmd_line() reaches it.
    return mode if mode in _MODES else "condensed"


def setup_mpi_console(rank: int, nproc: int, mode: Optional[str] = None) -> str:
    """Install the MPI console for this rank and return the mode that was applied.

    Idempotent, so the problem may re-apply it once ``--mpi-output`` has been parsed properly.
    """
    global _mpi_rank, _mpi_nproc, _mpi_mode
    _mpi_rank, _mpi_nproc = rank, nproc
    _mpi_mode = resolve_mpi_output_mode(mode)
    # Same policy on the C++ side, for oomph-lib's oomph_info output.
    _pyoomph._setup_mpi_console(rank, nproc, _mpi_mode)
    if mpi_console_is_filtering():
        _install_wrappers()
    return _mpi_mode
