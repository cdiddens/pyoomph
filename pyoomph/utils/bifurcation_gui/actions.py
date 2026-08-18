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

"""Commands of the bifurcation GUI, and the keyboard shortcuts that reach them.

Menus, toolbar buttons and key bindings are all generated from one list of :py:class:`Action`
records. That is deliberate: the previous version dispatched keys through a 200-line ``if
event.key==...`` chain, and adding a menu on top of it would have duplicated every command.

Accelerators are written the way the old key interface spelled them (``"space"``, ``"shift+space"``,
``"pagedown"``, ``"A"``), so muscle memory and any documentation of the old keys carry over. They
are matched against key events at dispatch time rather than bound individually through Tk, which is
what makes rebinding to an arbitrary key a one-line dictionary change.
"""

from dataclasses import dataclass
import json
import os
import sys

from ...typings import *


@dataclass
class Action:
    """One user-visible command."""

    id:str
    label:str
    callback:Callable[[],None]
    #: Where in the menu bar it appears, e.g. ``("Continuation",)`` or ``("Points","Tag")``.
    menu:tuple[str,...] | None=None
    #: Short label if it should also sit on the toolbar.
    toolbar:str | None=None
    tooltip:str=""
    #: "command", "check" or "separator-before" grouping hint.
    kind:str="command"
    #: Returns False when the command cannot currently be applied, so it can be greyed out.
    enabled_when:Callable[[],bool] | None=None
    #: For check items: reads and writes the underlying flag.
    getter:Callable[[],bool] | None=None
    #: Commands that touch the solver are dispatched through the controller's task runner.
    is_solver_task:bool=False


# ---------------------------------------------------------------------- key naming

# Tk spells keys as X11 keysyms; the accelerators use the matplotlib-ish names the tool has always
# used. Only keys whose keysym is not already the accelerator need an entry here.
_KEYSYM_TO_NAME={
    "space":"space", "Return":"enter", "KP_Enter":"enter", "BackSpace":"backspace",
    "Delete":"delete", "Escape":"escape", "Prior":"pageup", "Next":"pagedown",
    "Home":"home", "End":"end", "Tab":"tab",
    "Left":"left", "Right":"right", "Up":"up", "Down":"down",
    "asterisk":"*", "plus":"+", "minus":"-", "slash":"/", "equal":"=",
    "period":".", "comma":",", "less":"<", "greater":">", "underscore":"_",
    "KP_Multiply":"*", "KP_Add":"+", "KP_Subtract":"-", "KP_Divide":"/",
    "KP_Home":"home", "KP_End":"end", "KP_Prior":"pageup", "KP_Next":"pagedown",
}

_NAME_TO_DISPLAY={
    "space":"Space", "enter":"Enter", "backspace":"Backspace", "delete":"Delete",
    "escape":"Esc", "pageup":"PageUp", "pagedown":"PageDown", "home":"Home", "end":"End",
    "left":"Left", "right":"Right", "up":"Up", "down":"Down", "tab":"Tab",
}

#: Keysyms that are modifiers themselves - pressing them alone triggers nothing.
_MODIFIER_KEYSYMS={"Shift_L","Shift_R","Control_L","Control_R","Alt_L","Alt_R","Meta_L","Meta_R",
                   "Super_L","Super_R","Caps_Lock","Num_Lock","ISO_Level3_Shift"}


def event_to_accelerator(event)->str | None:
    """Turn a Tk key event into the accelerator spelling used by the keymap.

    Returns ``None`` for a bare modifier press. Shift is only spelled out for keys that do not
    already encode it in their keysym: pressing shift+a yields the keysym ``A``, so the accelerator
    is ``"A"`` (exactly as the old interface saw it), while shift+space yields ``"shift+space"``.
    """
    keysym=getattr(event,"keysym","")
    if not keysym or keysym in _MODIFIER_KEYSYMS:
        return None
    name=_KEYSYM_TO_NAME.get(keysym)
    encodes_shift=False
    if name is None:
        if len(keysym)==1:
            name=keysym
            encodes_shift=True   # 'A' vs 'a', '*' vs '8' - the character already carries shift
        else:
            name=keysym.lower()  # F1, Insert, ...
    state=int(getattr(event,"state",0) or 0)
    mods=[]
    if state & 0x4:
        mods.append("ctrl")
    if state & 0x8 or state & 0x20000:   # Mod1 on X11, Option on macOS
        mods.append("alt")
    if (state & 0x1) and not encodes_shift:
        mods.append("shift")
    return "+".join(mods+[name])


def format_accelerator(acc:str | None)->str:
    """Pretty form for a menu's accelerator column."""
    if not acc:
        return ""
    parts=acc.split("+")
    # A trailing "+" accelerator splits into ['',''] - the key itself is the plus sign.
    if acc.endswith("+") and len(parts)>=2:
        parts=parts[:-2]+["+"]
    out=[]
    for p in parts:
        if p in ("ctrl","alt","shift"):
            out.append(p.capitalize())
        elif p in _NAME_TO_DISPLAY:
            out.append(_NAME_TO_DISPLAY[p])
        elif len(p)>1:
            out.append(p.upper() if p[0]=="f" and p[1:].isdigit() else p.capitalize())
        elif p.isalpha():
            # An uppercase letter IS the shifted key ("A" is what shift+a produces), so it is shown
            # as such - otherwise "a" and "A", which are two different commands, would look alike.
            out.append(("Shift+" if p.isupper() else "")+p.upper())
        else:
            out.append(p)
    return "+".join(out)


# ---------------------------------------------------------------------- default bindings

#: Exactly the bindings of the previous key-driven interface.
DEFAULT_KEYMAP:dict[str,str]={
    "step":"space",
    "multistep":"shift+space",
    "step_shrink":"*",
    "ds_increase":"+",
    "ds_decrease":"-",
    "ds_reverse":"/",
    "arclength_scaling_on":"a",
    "arclength_scaling_off":"A",
    "locate_bifurcation":"b",
    "locate_pitchfork":"p",
    "transient_leave_0":"t",
    "transient_leave_1":"T",
    "toggle_splines":"i",
    "toggle_mode":"m",
    "toggle_move_point":"g",
    "split_branch":"x",
    "merge_branches":"X",
    "export_curves":"o",
    "cycle_observable":"y",
    "goto_selected":"enter",
    "delete_point":"backspace",
    "delete_point_alt":"delete",
    "select_prev":"pagedown",
    "select_next":"pageup",
    "select_first":"home",
    "select_last":"end",
    "abort":"escape",
}
for _d in range(10):
    DEFAULT_KEYMAP["tag_{:d}".format(_d)]=str(_d)


def _default_config_dir()->str:
    # Mirrors _default_cache_dir() in pyoomph/generic/jit_cache.py, but for configuration rather
    # than for cached build products.
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "pyoomph", "bifurcation_gui")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
        return os.path.join(base, "pyoomph", "bifurcation_gui")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
        return os.path.join(base, "pyoomph", "bifurcation_gui")


class KeyMap:
    """Action id to accelerator, persisted per user so a rebinding survives a restart."""

    def __init__(self,path:str | None=None) -> None:
        self.path=path if path is not None else os.path.join(_default_config_dir(),"keymap.json")
        self._map:dict[str,str]=dict(DEFAULT_KEYMAP)
        self.load()

    def load(self):
        # A broken or unreadable keymap must never keep the tool from starting; the defaults are
        # always a usable fallback.
        try:
            with open(self.path) as f:
                stored=json.load(f)
        except (OSError,ValueError):
            return
        if isinstance(stored,dict):
            for k,v in stored.items():
                if isinstance(v,str) and v:
                    self._map[str(k)]=v
                elif v is None:
                    self._map.pop(str(k),None)

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path),exist_ok=True)
            changed={k:v for k,v in self._map.items() if DEFAULT_KEYMAP.get(k)!=v}
            for k in DEFAULT_KEYMAP:
                if k not in self._map:
                    changed[k]=None #type:ignore[assignment]
            with open(self.path,"w") as f:
                json.dump(changed,f,indent=4,sort_keys=True)
            return True
        except OSError as e:
            print("Could not save the bifurcation GUI keymap to",self.path,":",e)
            return False

    def get(self,action_id:str)->str | None:
        return self._map.get(action_id)

    def action_for(self,accelerator:str)->str | None:
        for aid,acc in self._map.items():
            if acc==accelerator:
                return aid
        return None

    def set(self,action_id:str,accelerator:str | None):
        """Bind an action, clearing whatever else held that accelerator."""
        if accelerator is None:
            self._map.pop(action_id,None)
            return
        for aid,acc in list(self._map.items()):
            if acc==accelerator and aid!=action_id:
                del self._map[aid]
        self._map[action_id]=accelerator

    def reset_to_defaults(self):
        self._map=dict(DEFAULT_KEYMAP)

    def as_dict(self)->dict[str,str]:
        return dict(self._map)


from ...typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
