from __future__ import annotations
#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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
 
from .. import Expression
from .. import _pyoomph_core as _pyoomph 
import numpy
from ..typings import *


def split_numerical_text_header(header_line: str) -> "list[str] | None":
    """The column names of a pyoomph text file header, or None if this is not one.

    Split on TABS, which is what the writers join with, falling back to whitespace for a file that
    has none - the same rule LoadedTextDataFile uses, and for the same reason: a column name written
    before UNIT_SEPARATOR_IN_FILES existed can contain a space ("power[kg m^2/s^3]").
    """
    header = header_line.strip()
    if len(header) == 0 or header[0] != "#":
        return None
    body = header.strip("#").strip()
    names = body.split("\t") if "\t" in body else body.split()
    return [s.strip() for s in names if s.strip()]


def trim_numerical_text_file(filename: str, keep_until_time: float, time_column: str = "time",
                             rel_tolerance: float = 1e-9) -> "str | None":
    """Cut a pyoomph text file back to the last row at ``keep_until_time``.

    This is what a ``--runmode c`` that resumes from an earlier state needs: the rows the aborted run
    wrote after that instant describe a future that is about to be recomputed, and leaving them makes
    the time column of the file jump backwards in the middle.

    Returns None when the file was trimmed (or needed no trimming), or a human-readable reason why it
    could not be - the caller decides whether that is fatal. Nothing is written in that case.

    The surviving rows are copied **verbatim**, not reformatted: going through numpy would rewrite
    every number, and a resumed file is supposed to be indistinguishable from an uninterrupted one.
    The cut is after the LAST row at the resume time, so that a stationary or continuation run, which
    writes several rows at the same time value, keeps all of them.
    """
    import os
    if not os.path.isfile(filename):
        return None
    with open(filename, "r") as f:
        lines = f.readlines()
    if len(lines) == 0:
        return None
    names = split_numerical_text_header(lines[0])
    if names is None:
        return "it has no '#' header line"
    col = None
    for i, n in enumerate(names):
        if n == time_column or n.startswith(time_column + "["):
            col = i
            break
    if col is None:
        return ("it has no '" + time_column + "' column (its columns are: " + ", ".join(names) + ")")

    # The index of the last line that is still part of the run being resumed. Lines that do not parse
    # as data - a blank line, a second comment - belong with the row above them and are kept with it.
    keep_until = 0
    tol = rel_tolerance * max(abs(keep_until_time), 1.0)
    for i in range(1, len(lines)):
        fields = lines[i].split("\t") if "\t" in lines[i] else lines[i].split()
        if len(fields) <= col:
            continue
        try:
            t = float(fields[col])
        except ValueError:
            continue
        if t <= keep_until_time + tol:
            keep_until = i
    if keep_until == len(lines) - 1:
        return None  # nothing past the resume point

    # Via a temporary file in the same directory, so that an interruption cannot leave a half file
    # where a complete one was.
    tmpname = filename + ".trimming"
    with open(tmpname, "w") as f:
        f.writelines(lines[:keep_until + 1])
    os.replace(tmpname, filename)
    return None


class NumericalTextOutputFile:
    """A tab-separated text file of scalar rows, e.g. an observable over time.

    Under ``mpirun`` only rank 0 writes. What goes into such a file are quantities of the whole
    problem, so every rank producing the same row into the same file gains nothing and loses the
    file: the writes interleave *mid-number*, turning ``10.0\\t0.851169`` into
    ``10.0\\t0.8511691410.0\\t0.851169``. The other ranks keep a working object whose writes are
    dropped, so the same script runs serially and distributed without asking about the rank.

    Pass ``only_on_rank_zero=False`` if the rows really are per-rank - then give each rank its own
    file name, or they will overwrite each other in exactly the way described above.
    """

    def __init__(self, filename: str, open_mode: str = "w",header:list[str] | None=None,only_on_rank_zero:bool=True):
        from ..generic.mpi import get_mpi_nproc, get_mpi_rank
        self.filename = filename
        #: Whether this rank is the one that writes. See the class docstring.
        self.writes_here = (get_mpi_nproc() <= 1) or (not only_on_rank_zero) or (get_mpi_rank() == 0)
        self.file = None
        if self.writes_here:
            f = open(filename, open_mode)
            if f is None:
                raise RuntimeError("Could not open file "+str(filename))
            self.file = f
        self._closed = False
        if header:
            self.header(*header)

    def _write(self, line: str) -> None:
        if not self.writes_here:
            return
        if self._closed or self.file is None:
            raise RuntimeError("File was closed before")
        self.file.write(line)
        self.file.flush()

    def add_row(self, *args: float | Any):
        def params_to_float(p):
            if isinstance(p,(Expression,_pyoomph.GiNaC_GlobalParam)):
                return float(p)
            else:
                return p
        if len(args)==1 and isinstance(args[0],(list,tuple)):
            strargs = map(str, map(params_to_float,args[0]))
        else:
            strargs = map(str, map(params_to_float,[*args]))
        self._write("\t".join(strargs)+"\n")

    def header(self, *args: float | str | Any):
        self._write("#"+("\t".join(map(str, [*args]))) + "\n")

    def close(self):
        if self._closed:
            raise RuntimeError("File was already closed before")
        self._closed = True
        if self.file is not None:
            self.file.close()
            self.file = None

    def flush(self) -> None:
        if self._closed:
            raise RuntimeError("File was already closed before")
        if self.file is not None:
            self.file.flush()


class LoadedTextDataFile:
    """
    A wrapper to load pyoomph's text files including the header. This class serves as numpy.array, but also is aware of the header.
    You can still use it as numpy.array directly (or alternatively access its ``data`` member), but you can also directly access e.g.
    
        data=LoadedTextDataFile("my_file.txt")
        data[:,"velocity_x"]  # get the column with name starting with "velocity_x"
        data["velocity_x"]  # same as above, i.e. it is a column access, not a row access when used with a single string
        data["param"]           # get the parameter value of "param"
        
        data.get_column_index("velocity_x")  # get the column index of the column with name starting with "velocity_x"
        
    """
    def __init__(self, filename: str) -> None:
        try:
            f = open(filename, "r")
        except:
            raise RuntimeError("Cannot open the file '"+str(filename)+"'")
        header = f.readline().strip()
        f.close()
        if len(header) == 0 or header[0] != "#":
            raise RuntimeError("Found no header in the file "+str(filename))
                
        self.data: NPFloatArray = numpy.loadtxt(filename, ndmin=2)  # type:ignore
        # Split the header on TABS, which is what pyoomph joins it with, not on arbitrary whitespace.
        # A column name written today contains no space - units go into a file with
        # UNIT_SEPARATOR_IN_FILES between their symbols, e.g. "power[kg*m^2/s^3]" - but one written
        # before that does ("power[kg m^2/s^3]"), and whitespace-splitting tore such a name into
        # several tokens, which put every following name on the wrong column and offered the surplus
        # tokens up as parameters, where they raised an IndexError. Files written elsewhere may still
        # be space-separated, so fall back to that when there is no tab at all.
        header_body=header.strip().strip("#").strip()
        header_names=header_body.split("\t") if "\t" in header_body else header_body.split()
        header_names=[s.strip() for s in header_names if s.strip()]
        # Trailing "@key=value" entries never contain a space, so split them on whitespace too: they
        # are commonly appended to the last header field instead of behind a tab of their own.
        header_keys=[s.lstrip("@") for f in header_names[self.data.shape[1]:] for s in f.split()]
        for s in header_keys:
            if "=" not in s:
                raise RuntimeError("The header of the file '"+str(filename)+"' has more entries than "
                                   "the "+str(self.data.shape[1])+" columns of data, and '"+s+"' is "
                                   "not a '@key=value' parameter either")
        self.params={s.split("=")[0]:s.split("=",1)[1] for s in header_keys}
        self.columns=header_names[:self.data.shape[1]]
        self.access_params_via_brackets=True
                    
        

    @overload
    def get_column_index(self, index_or_name_start: list[str | int] | tuple[str | int, ...], exact_name: bool = False) -> NPIntArray: ...

    @overload
    def get_column_index(self, index_or_name_start: str | int, exact_name: bool = False) -> int: ...

    def get_column_index(self, index_or_name_start: list[str | int] | tuple[str | int, ...] | str | int, exact_name: bool = False) -> int | NPIntArray:
        if isinstance(index_or_name_start, (list, tuple)):
            rs: list[int] = []
            for i in index_or_name_start:
                rs.append(self.get_column_index(i, exact_name=exact_name))
            return numpy.array(rs, dtype=numpy.int32)

        if isinstance(index_or_name_start, str):
            # Find a unique column
            index = None
            for i, d in enumerate(self.columns):
                if (exact_name and d == index_or_name_start) or (not exact_name and d.startswith(index_or_name_start)):
                    if index is None:
                        index = i
                    else:
                        raise RuntimeError(
                            "At least two columns where found by the identifier '"+index_or_name_start+"'")
            if index is None:
                raise RuntimeError(
                    "Could not find a column beginning with the identifier '"+index_or_name_start+"'")
        else:
            index = index_or_name_start
            
        return index

    def get_column_data(self, index_or_name_start: list[str | int] | tuple[str | int, ...] | str | int, exact_name: bool = False) -> NPFloatArray:
        index=self.get_column_index(index_or_name_start, exact_name=exact_name)
        return self.data[:, index]  # type:ignore


    # The key here mirrors numpy's own flexible __getitem__/__setitem__ key argument
    # (int, str column name, slice, list/tuple of any of those, nested arbitrarily),
    # so it is genuinely dynamically typed rather than a typing gap to close.
    def _translate(self, key:Any) -> Any:
        if isinstance(key, str):
            return self.get_column_index(key)

        if isinstance(key, slice):
            start = self._translate(key.start) if key.start is not None else None
            stop = self._translate(key.stop) + 1 if key.stop is not None else None
            return slice(start, stop, key.step)

        if isinstance(key, list):
            return [self._translate(k) for k in key]

        if isinstance(key, tuple):
            return tuple(self._translate(k) for k in key)

        return key

    def __getitem__(self, key:Any) -> Any:
        if isinstance(key,str) and self.access_params_via_brackets and key in self.params:
            return self.params[key]
        if not isinstance(key, tuple):
            if isinstance(key, (str, list, slice)):
                key = (slice(None), key)
        return self.data[self._translate(key)]

    def __setitem__(self, key:Any, value:Any) -> None:
        if isinstance(key,str) and self.access_params_via_brackets and key in self.params:
            self.params[key]=value
            return
        if not isinstance(key, tuple):
            if isinstance(key, (str, list, slice)):
                key = (slice(None), key)
        self.data[self._translate(key)] = value

    def __getattr__(self, name:str) -> Any:
        return getattr(self.data, name)

    def __array__(self, dtype:Any=None) -> NPFloatArray:
        return numpy.asarray(self.data, dtype=dtype)


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
