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

# LoadedTextDataFile has to read back what pyoomph's text output writes. Since compound units are
# written with their symbols separated ("power[kg m^2/s^3]"), a column name can contain a space, and
# splitting the header on whitespace tore one name into several tokens: the following names ended up
# on the wrong columns and the surplus tokens were taken for "@key=value" parameters, where they
# raised an IndexError. The header is tab-joined, so it has to be split on tabs.

import pytest

from pyoomph.expressions.units import unit_to_string, kilogram, meter, second, kelvin
from pyoomph.utils.num_text_out import LoadedTextDataFile, NumericalTextOutputFile


def _write(tmp_path, header, rows, sep="\t"):
    fn = str(tmp_path/"data.txt")
    with open(fn, "w") as f:
        f.write("#"+sep.join(header)+"\n")
        for r in rows:
            f.write("\t".join(map(str, r))+"\n")
    return fn


ROWS = [(0.0, 1.0, 2.0), (1.0, 3.0, 4.0)]


def test_columns_with_spaced_units_are_read_back(tmp_path):
    """The case the output actually writes: units whose numerator or denominator has two symbols."""
    header = ["time[s]", "power[kg m^2/s^3]", "cp[m^2/(K s^2)]"]
    data = LoadedTextDataFile(_write(tmp_path, header, ROWS))
    assert data.columns == header
    assert not data.params
    assert list(data["power"]) == [1.0, 3.0]
    assert list(data["cp"]) == [2.0, 4.0]
    assert data.get_column_index("cp") == 2


def test_the_unit_strings_the_output_writes_do_contain_spaces():
    """Guards the premise: if unit_to_string ever stopped separating symbols, the test above would
    keep passing while testing nothing."""
    assert " " in unit_to_string(1.0*kilogram*meter**2/(kelvin*second**3), estimate_prefix=False)
    assert " " in unit_to_string(1.0*meter**2/(kelvin*second**2), estimate_prefix=False)


def test_parameters_are_still_picked_up(tmp_path):
    """Header entries beyond the columns are "@key=value" parameters, whether they sit behind a tab
    of their own or are appended to the last column name."""
    for sep, extra in ((("\t"), ["@Bo=0.5", "@mode=1"]), (("\t"), ["@Bo=0.5 @mode=1"])):
        fn = _write(tmp_path, ["time[s]", "power[kg m^2/s^3]", "cp[m^2/(K s^2)]"]+extra, ROWS, sep=sep)
        data = LoadedTextDataFile(fn)
        assert data.columns == ["time[s]", "power[kg m^2/s^3]", "cp[m^2/(K s^2)]"]
        assert data.params == {"Bo": "0.5", "mode": "1"}
        assert data["Bo"] == "0.5"


def test_space_separated_headers_still_work(tmp_path):
    """Files not written by pyoomph may have no tab in the header at all."""
    fn = _write(tmp_path, ["time", "power", "cp"], ROWS, sep=" ")
    data = LoadedTextDataFile(fn)
    assert data.columns == ["time", "power", "cp"]
    assert list(data["power"]) == [1.0, 3.0]


def test_a_surplus_header_entry_that_is_no_parameter_is_reported(tmp_path):
    fn = _write(tmp_path, ["time", "power", "cp", "leftover"], ROWS)
    with pytest.raises(RuntimeError, match="leftover"):
        LoadedTextDataFile(fn)


def test_round_trip_through_the_writer(tmp_path):
    """End to end: NumericalTextOutputFile writes the header, LoadedTextDataFile reads it."""
    fn = str(tmp_path/"out.txt")
    out = NumericalTextOutputFile(fn, header=["time[s]", "power[kg m^2/s^3]"])
    out.add_row(0.0, 1.0)
    out.add_row(1.0, 3.0)
    out.close()
    data = LoadedTextDataFile(fn)
    assert data.columns == ["time[s]", "power[kg m^2/s^3]"]
    assert list(data["power"]) == [1.0, 3.0]
