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

"""A ``:vidtime:`` role that links into an embedded video at a given time.

The tutorial embeds its videos as raw ``<figure id="...">`` HTML blocks (only in the HTML build).
This role turns a piece of text into a link that seeks the video inside such a figure to a
timestamp, scrolls it into view and focuses it, all without leaving the page - see
``_static/js/vidtime.js`` for the client side.

Usage::

    :vidtime:`vidbifgui1#1:23`                    -> link labelled "1:23"
    :vidtime:`branch switching <vidbifgui1#1:23>` -> link with your own label

The timestamp is ``[hh:]mm:ss[.fff]`` or plain seconds. Builders other than HTML have no video to
jump to, so there the role degrades to its plain label text.
"""

from docutils import nodes
from docutils.parsers.rst.states import Inliner
from sphinx.util.docutils import SphinxRole
from sphinx.util import logging
import os
import re

logger = logging.getLogger(__name__)


class vidtime_ref(nodes.Inline, nodes.TextElement):
    """Carries the target figure id and the time in seconds until the writer runs."""


def _parse_time(text: str) -> float:
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError("too many ':' separated parts")
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + float(p)
    return seconds


class VidTimeRole(SphinxRole):
    _explicit = re.compile(r"^(?P<label>.+?)\s*<(?P<target>[^<>]+)>$", re.DOTALL)

    def run(self):
        text = nodes.unescape(self.text)
        m = self._explicit.match(text)
        label = None
        if m:
            label, text = m.group("label"), m.group("target")
        if "#" not in text:
            msg = self.inliner.reporter.error(
                "vidtime needs 'figure-id#time', got %r" % text, line=self.lineno)
            return [self.inliner.problematic(self.rawtext, self.rawtext, msg)], [msg]
        target, timestr = text.rsplit("#", 1)
        try:
            seconds = _parse_time(timestr.strip())
        except ValueError:
            msg = self.inliner.reporter.error(
                "vidtime cannot read the timestamp %r (use [hh:]mm:ss or seconds)" % timestr,
                line=self.lineno)
            return [self.inliner.problematic(self.rawtext, self.rawtext, msg)], [msg]
        node = vidtime_ref("", label if label is not None else timestr.strip())
        node["vidtime_target"] = target.strip()
        node["vidtime_seconds"] = seconds
        return [node], []


def visit_vidtime_html(self, node):
    self.body.append(
        '<a class="vidtime" href="#%s" data-vidtime-target="%s" data-vidtime-seconds="%g">'
        % (self.encode(node["vidtime_target"]), self.encode(node["vidtime_target"]),
           node["vidtime_seconds"]))


def depart_vidtime_html(self, node):
    self.body.append("</a>")


def visit_vidtime_plain(self, node):
    pass


def depart_vidtime_plain(self, node):
    pass


def setup(app):
    app.add_node(vidtime_ref,
                 html=(visit_vidtime_html, depart_vidtime_html),
                 latex=(visit_vidtime_plain, depart_vidtime_plain),
                 text=(visit_vidtime_plain, depart_vidtime_plain),
                 man=(visit_vidtime_plain, depart_vidtime_plain),
                 texinfo=(visit_vidtime_plain, depart_vidtime_plain))
    app.add_role("vidtime", VidTimeRole())
    app.add_js_file("js/vidtime.js")
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
