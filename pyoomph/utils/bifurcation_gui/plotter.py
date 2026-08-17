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

"""Draws a bifurcation diagram onto a matplotlib figure.

The figure is constructed through the object-oriented API (``matplotlib.figure.Figure``) rather than
through ``pyplot``. That is what makes the tool independent of the selected matplotlib backend: the
user interface embeds this figure in its own canvas, so importing ``pyoomph.output.plotting``
(which forces the ``Agg`` backend) before or after this module no longer matters. The previous
pyplot-based version had to set ``PYOOMPH_MPLBACKEND`` at import time and raised a RuntimeError
telling the user to reorder their imports when it lost that race.
"""

import matplotlib.text
from matplotlib.figure import Figure

from pathlib import Path
import numpy
import os

from ...typings import *


class BifurcationDiagramPlotter:
    """Owns the matplotlib figure and renders the diagram held by a controller.

    Also serves as the controller's :py:class:`~pyoomph.utils.bifurcation_gui.controller.BifurcationViewLimits`,
    since the visible range is what the multistep stop criterion and the point-insertion metric are
    defined against.
    """

    def __init__(self,figsize=(8,6),dpi=100) -> None:
        self.figure=Figure(figsize=figsize,dpi=dpi)
        self.axes=self.figure.add_subplot(1,1,1)
        self.figure.set_layout_engine("constrained")
        self._view_initialised=False
        #: Called after each draw so an embedding canvas can blit; set by the user interface.
        self.on_drawn:Callable[[],None] | None=None

    # ---------------------------------------------------------------- view limits interface

    def get_xlim(self): return self.axes.get_xlim()
    def get_ylim(self): return self.axes.get_ylim()
    def get_xscale(self): return self.axes.get_xscale()
    def get_yscale(self): return self.axes.get_yscale()

    def set_xlim(self,lo,hi): self.axes.set_xlim(lo,hi)
    def set_ylim(self,lo,hi): self.axes.set_ylim(lo,hi)
    def set_xscale(self,scale): self.axes.set_xscale(scale)
    def set_yscale(self,scale): self.axes.set_yscale(scale)

    def initialise_view(self,controller):
        """Set the starting window once, from the explicit initial view or around the first point.

        The tiny 1e-4 box around the first point is deliberate: the multistep sweep stops at the
        axes border, so a fresh diagram starts zoomed in and the user zooms out to sweep further.
        """
        if self._view_initialised:
            return
        self._view_initialised=True
        cp=controller._get_current_point().get_coordinate(controller._get_current_observable())
        if controller._initial_view is not None:
            self.axes.set_xlim(controller._initial_view[0],controller._initial_view[1])
            self.axes.set_ylim(controller._initial_view[2],controller._initial_view[3])
        else:
            self.axes.set_xlim(cp[0]-1e-4,cp[0]+1e-4)
            self.axes.set_ylim(cp[1]-1e-4,cp[1]+1e-4)

    def apply_saved_view(self,fullinfo:dict):
        """Restore limits and scales from a state.json dict."""
        self.axes.set_xlim(fullinfo["xlim"])
        self.axes.set_ylim(fullinfo["ylim"])
        self.axes.set_xscale(fullinfo["xscale"])
        self.axes.set_yscale(fullinfo["yscale"])
        self._view_initialised=True

    def autoscale_to_data(self,controller,margin=0.05):
        rng=controller.data_range()
        if rng is None:
            return
        (xmin,xmax),(ymin,ymax)=rng
        dx=(xmax-xmin) or max(abs(xmax),1e-4)
        dy=(ymax-ymin) or max(abs(ymax),1e-4)
        self.axes.set_xlim(xmin-margin*dx,xmax+margin*dx)
        self.axes.set_ylim(ymin-margin*dy,ymax+margin*dy)

    # ---------------------------------------------------------------- drawing

    def _clear_artists(self):
        gca=self.axes
        while len(gca.lines)>0:
            gca.lines[0].remove()
        while len(gca.artists)>0:
            gca.artists[0].remove()
        annotations = [child for child in gca.get_children() if isinstance(child, matplotlib.text.Annotation)]
        for a in annotations:
            a.remove()
        # Note: gca.texts (an ArtistList) does not support item deletion, so the previous
        # "except: del gca.texts[-1]" fallback would itself raise TypeError if .remove() ever
        # failed. Removing from a snapshot list and ignoring individual failures is safe.
        for t in list(gca.texts):
            try:
                t.remove()
            except Exception:
                pass

    def draw(self,controller,infotext:str | None=None):
        """Redraw the whole diagram. ``infotext`` is the centred "busy" box."""
        if controller.current_point is None or controller._current_observable is None:
            return
        self.initialise_view(controller)

        gca=self.axes
        observable=controller._current_observable
        xlim=list(gca.get_xlim())
        ylim=list(gca.get_ylim())
        xscal=gca.get_xscale()
        yscal=gca.get_yscale()

        def extend_lims(p):
            xlim[0]=min(xlim[0],p[0])
            xlim[1]=max(xlim[1],p[0])
            ylim[0]=min(ylim[0],p[1])
            ylim[1]=max(ylim[1],p[1])

        self._clear_artists()

        for b in controller.branches:
            color="red" if b == controller.current_branch else "grey"

            if controller.interpolated_splines:
                segs,stabs=b.smooth_branch_stab_list(observable)
            else:
                segs,stabs=b.to_branch_stab_list(observable)

            for seg,stab in zip(segs,stabs):
                if stab == True:
                    dt="-"
                    lw=1.5
                elif stab == False:
                    dt="dashed"
                    lw=0.75
                else:
                    dt="dotted"
                    lw=1.0
                gca.plot(seg[:,0],seg[:,1], linestyle=dt,color=color,linewidth=lw)
            normpts=numpy.array([p.get_coordinate(observable) for p in b if p.eig_value_Re!=0],ndmin=2)
            if len(normpts)>0:
                gca.plot(normpts[:,0],normpts[:,1], 'o', markersize=3,color=color)
            for p in b:
                if p.eig_value_Re==0:
                    pc=p.get_coordinate(observable)
                    gca.plot([pc[0]],[pc[1]], marker='o', markersize=6,color="brown")
                    if p.bifurcation_info is not None:
                        shorts={"transcritical":"T","fold":"F","pitchfork":"P","Hopf":"H"}
                        if p.bifurcation_info["type"] in shorts:
                            gca.annotate(str(p.tag)+"," if p.tag>=0 else ""+ shorts[p.bifurcation_info["type"]],(pc[0],pc[1]))
                        elif p.tag>=0:
                            gca.annotate(str(p.tag),(pc[0],pc[1]))
                elif p.tag>=0:
                    pc=p.get_coordinate(observable)
                    gca.annotate(str(p.tag),(pc[0],pc[1]))

        if controller.current_point is not None:
            extend_lims(controller.current_point.get_coordinate(observable))
            pc=controller.current_point.get_coordinate(observable,with_eigen=True)
            if controller._mode=="al":
                gca.plot([pc[0]],[pc[1]], marker='o', markersize=5,color="green" )
                tang=controller._tangs.get(observable)
                if tang is not None and controller._last_ds is not None:
                    x0=numpy.array([pc[0],pc[1]])
                    dx=controller._last_ds*tang
                    extend_lims(x0)
                    xy_end=(float(x0[0]+dx[0]),float(x0[1]+dx[1]))
                    xy_start=(float(x0[0]),float(x0[1]))
                    gca.annotate("", xy=xy_end, xytext=xy_start,arrowprops=dict(arrowstyle="->"),annotation_clip=False)
                    for i,bst in enumerate(controller.current_point._branch_switch_tangs):
                        dx=controller._last_ds*bst[observable]
                        extend_lims(x0)
                        xy_end=(float(x0[0]+dx[0]),float(x0[1]+dx[1]))
                        xy_start=(float(x0[0]),float(x0[1]))
                        gca.annotate("", xy=xy_end, xytext=xy_start,arrowprops=dict(arrowstyle="->",color="brown",linewidth=1 if i==0 else 0.1),annotation_clip=False)
                eigv=pc[2]+1j*pc[3]
                pttext="({:3.3g},{:3.3g})\n".format(pc[0],pc[1])+f'{eigv:.2g}'
            else:
                pttext=None
        else:
            pttext=None

        if controller.selected_point is not None:
            pc=controller.selected_point.get_coordinate(observable,with_eigen=True)
            if controller._mode=="mp" and controller._move_point:
                gca.plot([pc[0]],[pc[1]], marker='o', markersize=5,color="blue")
            else:
                gca.plot([pc[0]],[pc[1]], marker='x', markersize=12,color="grey")
            eigv=pc[2]+1j*pc[3]
            seltext="({:3.3g},{:3.3g})\n".format(pc[0],pc[1])+f'{eigv:.2g}'
        else:
            seltext=None

        if controller.parameter_range is not None and len(controller.parameter_range)==2:
            gca.set_xlim(controller.parameter_range[0],controller.parameter_range[1])

        gca.set_xlabel(controller.get_bifurcation_parameter().get_name())
        gca.set_ylabel(observable)
        gca.set_xlim(xlim[0],xlim[1])
        gca.set_ylim(ylim[0],ylim[1])
        gca.set_xscale(xscal)
        gca.set_yscale(yscal)
        if infotext is not None:
            gca.text(0.5, 0.5, infotext,horizontalalignment='center',verticalalignment='center',transform=gca.transAxes,bbox = dict(boxstyle="round", fc="lightgrey", ec="0.5", alpha=0.9))

        if pttext is not None:
            gca.text(0.1, 1.01, pttext,horizontalalignment='center',verticalalignment='bottom',transform=gca.transAxes,color="red")
        if seltext is not None:
            gca.text(0.9, 1.01, seltext,horizontalalignment='center',verticalalignment='bottom',transform=gca.transAxes,color="grey")

        if self.on_drawn is not None:
            self.on_drawn()

        if controller._out_demo_video:
            ddir=controller.problem.get_output_directory(controller.data_subdir)
            odir=os.path.join(ddir,"demo_movie")
            Path(odir).mkdir(parents=True,exist_ok=True)
            self.figure.savefig(os.path.join(odir,"plot_{:06d}.png".format(controller._demo_video_step)))
            controller._demo_video_step+=1

    def savefig(self,fname:str,**kwargs):
        self.figure.savefig(fname,**kwargs)


from ...typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
