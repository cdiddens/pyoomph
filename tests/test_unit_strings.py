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

# unit_to_string(value) -> (unit, factor, multiplier), where factor*multiplier is the number to print
# in front of that unit. The property that matters is that the PRINTED quantity equals the input:
# "1 um^2" must not be handed back for 1e-6 m^2, which is a million times larger.
#
# The prefix-estimating branch had no coverage at all - every call in pyoomph passes
# estimate_prefix=False - and it was wrong in four separate ways, all found once the bifurcation GUI
# started using it to report eigenvalues in 1/s and observables in their own units.

import pytest

from pyoomph.expressions.units import (unit_to_string, meter, second, kilogram, kelvin, newton,
                                       pascal, watt, volt, farad, milli, kilo)


# (input, expected unit string, expected printed number). Each expectation is checkable by hand:
# 1 mm^2 IS 1e-6 m^2, 1 1/Mm IS 1e-6 /m, and so on.
CASES = [
    # Plain lengths, including the prefixes that did not exist (pico, tera).
    (7e-4*meter,        "mm",    0.7),
    (0.5*meter,         "m",     0.5),
    (3e-13*meter,       "pm",    0.3),
    (7e12*second,       "Ts",    7.0),
    # POWERS. The prefix is written in front of the whole numerator, so it binds to the exponent too:
    # "mm^2" means (mm)^2 = 1e-6 m^2. Treating the exponent as 1 reported 1e-6 m^2 as "1 um^2".
    (1e-6*meter**2,     "mm^2",  1.0),
    (1e-12*meter**2,    "um^2",  1.0),
    (1.0*meter**2,      "m^2",   1.0),
    (1e-9*meter**3,     "mm^3",  1.0),
    # Inverse units put the prefix in the denominator, inverted.
    (1e-6/meter,        "1/Mm",  1.0),
    (1e9/meter,         "1/nm",  1.0),
    # Rates: the case the bifurcation GUI reports eigenvalues with. NEGATIVE values used to match the
    # smallest prefix on the first comparison, so -0.5/s was printed as "1/Gs".
    (-0.5/second,       "1/s",  -0.5),
    (-5.0/second,       "1/s",  -5.0),
    # kilogram already carries a prefix of its own.
    (2.0*kilogram,      "kg",    2.0),
    (1e-6*kilogram,     "mg",    1.0),
    # Derived units keep their compound name.
    (1e5*pascal,        "MPa",   0.1),
    (1e-7*newton,       "uN",    0.1),
    (1.0*watt,          "W",     1.0),
    (1e-7*watt,         "uW",    0.1),
    (1.0*volt,          "V",     1.0),
    (1e-3*volt,         "mV",    1.0),
    (1.0*farad,         "F",     1.0),
    (1e-12*farad,       "pF",    1.0),
]


# Compound units that no derived name covers, so contrib_part has to spell them out.
SPELLED_OUT = [
    (1.0*kilogram*meter**2/second**3/kelvin, "kg m^2/(K s^3)"),
    (1.0*meter/second,                       "m/s"),
    (1.0*meter/second**2,                    "m/s^2"),
    (1.0*kilogram/(meter*second**2),         "Pa"),
    (1.0/second,                             "1/s"),
    (1.0*kilogram*meter,                     "kg m"),
]


@pytest.mark.parametrize("value,expected", SPELLED_OUT)
def test_compound_units_are_separated_and_bracketed(value, expected):
    """Symbols are separated, and a denominator of several factors is bracketed.

    Both matter for reading: run together, "kg/ms^2" looks like kg per millisecond squared, and even
    separated, "kg/m s^2" reads as kg*s^2/m - the reciprocal of what is meant.
    """
    assert unit_to_string(value, estimate_prefix=False) == expected
    # The prefix-estimating path must spell them the same way.
    assert unit_to_string(value)[0] == expected


@pytest.mark.parametrize("value,expected_unit,expected_number", CASES)
def test_unit_to_string_prints_the_quantity_it_was_given(value, expected_unit, expected_number):
    unit, factor, mult = unit_to_string(value)
    assert unit == expected_unit, "got '{:s}' for {!s:s}".format(unit, value)
    printed = factor*mult
    assert abs(printed - expected_number) < 1e-9*max(1.0, abs(expected_number)), \
        "printed {:.12g} {:s}, expected {:.12g}".format(printed, unit, expected_number)


def test_zero_and_dimensionless_have_no_prefix():
    """Zero used to match the smallest prefix, so 0 metres printed as "0 nm"."""
    unit, factor, mult = unit_to_string(0.0*meter)
    assert factor*mult == 0.0
    assert unit in ("", "m"), unit          # no unit survives 0*meter in GiNaC; either is honest
    unit, factor, mult = unit_to_string(2.0)
    assert unit == "" and factor*mult == 2.0


def test_estimate_prefix_false_is_untouched():
    """Every existing caller in pyoomph passes estimate_prefix=False and must keep its answer."""
    assert unit_to_string(7e-4*meter, estimate_prefix=False) == "m"
    assert unit_to_string(1e-6*meter**2, estimate_prefix=False) == "m^2"
    assert unit_to_string(-0.5/second, estimate_prefix=False) == "1/s"
    assert unit_to_string(1e5*pascal, estimate_prefix=False) == "Pa"
    assert unit_to_string(1.0*watt, estimate_prefix=False) == "W"
