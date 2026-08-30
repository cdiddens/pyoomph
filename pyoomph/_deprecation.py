from __future__ import annotations
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

"""Support for renaming keyword arguments without breaking existing scripts.

Nothing in here is part of the public API; it exists so that a rename is one decorator at the
definition rather than a hand-written check in every constructor.
"""

import functools
import warnings

from .typings import *


def deprecated_kwargs(**aliases: str) -> "Callable[[Callable[...,Any]],Callable[...,Any]]":
    """Accept renamed keyword arguments under their old names, with a :py:class:`DeprecationWarning`.

    Decorate with e.g. ``@deprecated_kwargs(coordinate_system="coordsys")`` and a call passing
    ``coordinate_system=...`` keeps working, is forwarded to ``coordsys`` and warns. Passing both
    spellings at once raises a :py:class:`TypeError` rather than silently preferring one of them.

    The decorated function keeps its own signature via :py:func:`functools.wraps`, so the new name
    is what documentation and type checkers see - which is the point, since the old one is not
    supposed to be discoverable any more.
    """
    def decorator(func:"Callable[...,Any]")->"Callable[...,Any]":
        @functools.wraps(func)
        def wrapper(*args:Any, **kwargs:Any)->Any:
            for old, new in aliases.items():
                if old not in kwargs:
                    continue
                if new in kwargs:
                    raise TypeError(f"{func.__qualname__}() got both '{new}' and its deprecated alias '{old}'. Pass only '{new}'.")
                warnings.warn(f"The argument '{old}' of {func.__qualname__}() is deprecated, use '{new}' instead.", DeprecationWarning, stacklevel=2)
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)
        setattr(wrapper, "__deprecated_kwargs__", dict(aliases))
        return wrapper
    return decorator


def deprecated_attribute_alias(old:str, new:str)->property:
    """A property that reads and writes the attribute ``new`` under its former name ``old``, warning.

    The counterpart of :py:func:`deprecated_kwargs` for the attribute a renamed constructor argument
    is stored in, since that is just as visible to user code as the argument itself. Assign it in the
    class body as ``coordinate_system = deprecated_attribute_alias("coordinate_system","coordsys")``.

    Do not use this for a name the base class assigns in its own ``__init__``: the assignment would go
    through this property and the base class would end up writing the aliased attribute instead of its
    own.
    """
    def _warn(owner:str)->None:
        warnings.warn(f"The attribute '{old}' of {owner} is deprecated, use '{new}' instead.", DeprecationWarning, stacklevel=3)
    def getter(self:Any)->Any:
        _warn(type(self).__name__)
        return getattr(self, new)
    def setter(self:Any, value:Any)->None:
        _warn(type(self).__name__)
        setattr(self, new, value)
    return property(getter, setter, doc=f"Deprecated alias of :py:attr:`{new}`.")
