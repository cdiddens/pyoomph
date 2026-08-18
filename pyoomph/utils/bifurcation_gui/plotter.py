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

from .model import AXIS_OBSERVABLE

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
        #: Draw branches from other slices of parameter space, faintly, for context. Turning this off
        #: hides them entirely; they are never drawn in the current diagram's colours either way.
        self.show_other_slices=True
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
        cp=controller._get_current_point().get_coordinate(controller.y_axis,xspec=controller.x_axis)
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

    def _draw_faint_branch(self,branch,xaxis,yaxis):
        """A branch from another slice: one washed-out line, no markers, no stability coding.

        Deliberately featureless - it is context, not data being read off.
        """
        segs,_=branch.to_branch_stab_list(yaxis,xspec=xaxis)
        for seg in segs:
            self.axes.plot(seg[:,0],seg[:,1],linestyle="-",color="0.82",linewidth=0.8,zorder=0)

    def _draw_locus_branch(self,branch,xaxis,yaxis,current=False):
        """A curve of bifurcation points: one line plus a marker per computed point.

        Drawn in the same brown the individual bifurcations use, so a fold curve reads as "these are
        all folds" rather than as another solution branch.
        """
        pts=numpy.array([p.get_coordinate(yaxis,xspec=xaxis) for p in branch],ndmin=2)
        if len(pts)==0:
            return
        self.axes.plot(pts[:,0],pts[:,1],linestyle="-",color="brown",
                       linewidth=2.0 if current else 1.0,alpha=1.0 if current else 0.55)
        self.axes.plot(pts[:,0],pts[:,1],marker="o",markersize=4 if current else 3,
                       linestyle="none",color="brown",alpha=1.0 if current else 0.55)
        for p in branch:
            if p.tag>=0:
                pc=p.get_coordinate(yaxis,xspec=xaxis)
                self.axes.annotate(str(p.tag),(pc[0],pc[1]))

    def draw(self,controller,infotext:str | None=None):
        """Redraw the whole diagram. ``infotext`` is the centred "busy" box."""
        if controller.current_point is None or controller._current_observable is None:
            return
        self.initialise_view(controller)

        gca=self.axes
        xaxis,yaxis=controller.x_axis,controller.y_axis
        # The tangent bookkeeping is keyed by observable name, so the arrows only apply when the
        # vertical axis IS that observable - on a parameter-vs-parameter plot they are meaningless.
        tang_key=yaxis[1] if yaxis[0]==AXIS_OBSERVABLE else None
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
            # A branch continued in a different parameter, or one from a file that never recorded the
            # parameter now on an axis, has nothing to show here. Skipped rather than drawn wrong.
            if not controller.branch_can_be_plotted(b):
                continue
            # A locus is checked BEFORE the slice: it varies two parameters on purpose, so it belongs
            # to no single slice and greying it out would hide the one curve that says where the
            # bifurcation sits for every value of the second parameter. Every point of it IS the
            # bifurcation, which is also why the stability segmentation is bypassed - it would see a
            # zero real part at every point and alternate the line style from one to the next.
            if b.kind=="locus":
                self._draw_locus_branch(b,xaxis,yaxis,current=b is controller.current_branch)
                continue
            # A branch from another slice of parameter space is a different physical result. It is
            # kept on screen for context but never in the current diagram's colours, and it is not
            # selectable - see BifurcationController.select_nearest_point.
            if not controller.branch_is_on_current_slice(b):
                if not self.show_other_slices:
                    continue
                self._draw_faint_branch(b,xaxis,yaxis)
                continue
            color="red" if b == controller.current_branch else "grey"

            if controller.interpolated_splines:
                segs,stabs=b.smooth_branch_stab_list(yaxis,xspec=xaxis,
                                                     trust_inferred=controller.trust_inferred_stability,
                                                     include_modes=controller.count_normal_modes_in_stability)
            else:
                segs,stabs=b.to_branch_stab_list(yaxis,xspec=xaxis,
                                                 trust_inferred=controller.trust_inferred_stability,
                                                 include_modes=controller.count_normal_modes_in_stability)

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
            # A bifurcation that quick mode BRACKETED rather than located: an open marker halfway
            # between the two points it lies between, so it cannot be mistaken for a computed one.
            for ip,p in enumerate(b):
                if p.detected_bifurcation is None or ip==0:
                    continue
                try:
                    a=b[ip-1].get_coordinate(yaxis,xspec=xaxis)
                    c2=p.get_coordinate(yaxis,xspec=xaxis)
                except KeyError:
                    continue
                gca.plot([0.5*(a[0]+c2[0])],[0.5*(a[1]+c2[1])],marker="o",markersize=9,
                         markerfacecolor="none",markeredgecolor="brown",markeredgewidth=1.4,
                         linestyle="none")
            normpts=numpy.array([p.get_coordinate(yaxis,xspec=xaxis) for p in b if p.eig_value_Re!=0],ndmin=2)
            if len(normpts)>0:
                gca.plot(normpts[:,0],normpts[:,1], 'o', markersize=3,color=color)
            for p in b:
                if p.eig_value_Re==0:
                    pc=p.get_coordinate(yaxis,xspec=xaxis)
                    gca.plot([pc[0]],[pc[1]], marker='o', markersize=6,color="brown")
                    if p.bifurcation_info is not None:
                        # get_normal_form writes "hopf" in lower case; keyed as "Hopf" the letter never
                        # appeared. And the tag used to swallow it: without the brackets, Python read
                        # this as (tag+",") if tagged else (""+letter), so a tagged bifurcation showed
                        # its number and an untagged one its letter, never both.
                        shorts={"transcritical":"T","fold":"F","pitchfork":"P","hopf":"H"}
                        if p.bifurcation_info["type"] in shorts:
                            gca.annotate((str(p.tag)+"," if p.tag>=0 else "")+shorts[p.bifurcation_info["type"]],(pc[0],pc[1]))
                        elif p.tag>=0:
                            gca.annotate(str(p.tag),(pc[0],pc[1]))
                elif p.tag>=0:
                    pc=p.get_coordinate(yaxis,xspec=xaxis)
                    gca.annotate(str(p.tag),(pc[0],pc[1]))

        if controller.current_point is not None:
            extend_lims(controller.current_point.get_coordinate(yaxis,xspec=xaxis))
            pc=controller.current_point.get_coordinate(yaxis,with_eigen=True,xspec=xaxis)
            if controller._mode=="al":
                gca.plot([pc[0]],[pc[1]], marker='o', markersize=5,color="green" )
                x0=numpy.array([pc[0],pc[1]])
                xy_start=(float(x0[0]),float(x0[1]))
                tang=controller._tangs.get(tang_key) if tang_key is not None else None
                if tang is not None and controller._last_ds is not None:
                    dx=controller._last_ds*tang
                    xy_end=(float(x0[0]+dx[0]),float(x0[1]+dx[1]))
                    extend_lims(xy_end)
                    gca.annotate("", xy=xy_end, xytext=xy_start,arrowprops=dict(arrowstyle="->"),annotation_clip=False)
                # Deliberately NOT nested in the block above. At a located bifurcation there is no
                # arclength tangent at all - _update_tangents empties _tangs there on purpose - and that
                # is precisely the point where these arrows are the only ones to draw. Nested under
                # "tang is not None" they could therefore never appear anywhere.
                # abs(ds): the arrows are directions off the bifurcation, which do not swap over when
                # the direction of travel is reversed. It undoes the scaling the controller applied.
                scale=abs(controller._last_ds) if controller._last_ds else 1.0
                # The heavy arrow is the one the default action would take RIGHT NOW: a branch switch
                # defaults its direction to the sign of ds, and index 0 is the +1 direction. The other
                # is drawn thin rather than left out, so a pitchfork does not look like it has one arm.
                primary=0 if (controller._last_ds is None or controller._last_ds>=0) else 1
                # A fold or a Hopf has no second steady branch: its two arrows are the +-eigenvector a
                # transient would leave along, so they are dashed to say "this is a perturbation, not a
                # branch", and neither of them is the preferred one.
                perturbing=controller.current_point._departure_kind=="perturb"
                for i,bst in enumerate(controller.current_point._departure_tangs if tang_key is not None else []):
                    if tang_key not in bst:
                        # An observable this direction was never evaluated for: it is recorded per
                        # observable, and the set can differ from the one the point was computed with.
                        # Worth no arrow, certainly not a KeyError out of the middle of a redraw.
                        continue
                    dx=scale*bst[tang_key]
                    xy_end=(float(x0[0]+dx[0]),float(x0[1]+dx[1]))
                    extend_lims(xy_end)
                    aprops=dict(arrowstyle="->",color="brown",
                                linewidth=1.0 if perturbing else (1.4 if i==primary else 0.6),
                                linestyle="dashed" if perturbing else "solid")
                    gca.annotate("", xy=xy_end, xytext=xy_start,arrowprops=aprops,annotation_clip=False)
                eigv=pc[2]+1j*pc[3]
                pttext="({:3.3g},{:3.3g})\n".format(pc[0],pc[1])+f'{eigv:.2g}'
            else:
                pttext=None
        else:
            pttext=None

        if controller.selected_point is not None:
            pc=controller.selected_point.get_coordinate(yaxis,with_eigen=True,xspec=xaxis)
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

        gca.set_xlabel(controller.axis_label(xaxis))
        gca.set_ylabel(controller.axis_label(yaxis))
        # The parameters held fixed are part of the result, so they go on the figure itself - a saved
        # PDF of a diagram that does not say what slice it is a section through cannot be captioned.
        slice_desc=controller.describe_current_slice()
        gca.set_title(slice_desc if slice_desc else "",fontsize="small",loc="right")
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
