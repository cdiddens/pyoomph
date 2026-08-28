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

"""Plotting of one-dimensional domains as ordinary x-y graphs.

:py:mod:`pyoomph.output.plotting` draws a two-dimensional mesh as a spatial map - a colour-keyed
field over the domain, with arrows and streamlines on top. A one-dimensional domain has no area to
fill; what one wants there is a graph, with visible axes, labels and a legend. That is what
:py:class:`MatplotlibPlotter1D` provides, with the same ``define_plot``/``add_plot`` shape as its 2d
counterpart, from which it inherits everything that is not about the geometry: file output,
eigenvector plotting, the MPI merge of a distributed mesh, and all the overlays (colorbars, text,
time labels).

A 1d domain need not live in 1d space. An interface of a 2d problem, or a ``LineMesh`` built with
``nodal_dimension=2``, carries a y coordinate as well, and its own shape in the plane is often the
interesting thing. Such a curve is drawn by plotting one coordinate against another, optionally
colour-coded by a field - see :py:meth:`MatplotlibPlotter1D.add_curve`.
"""

import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.ticker

from .plotting import (MatplotLibPart, MatplotLibAxes, MatplotLibLinePlot, MatplotLibColorbar,
                       MatplotlibPlotter, PlotTransform)
from ..expressions.generic import ExpressionNumOrNone, ExpressionOrNum
from ..expressions.units import unit_to_string
from ..meshes.ordering import sort_line_segments

from ..typings import *
import numpy

if TYPE_CHECKING:
    from ..generic.problem import Problem
    from ..meshes.meshdatacache import MeshDataEigenModes
    from ..meshes.ordering import SortAlongAxis


#: How a coordinate or pseudo-coordinate is spelled on an axis label.
_AXIS_PRETTY_NAMES = {"coordinate_x": "$x$", "coordinate_y": "$y$", "coordinate_z": "$z$",
                      "lagrangian_x": "$X$", "lagrangian_y": "$Y$", "lagrangian_z": "$Z$",
                      "arclength": "$s$", "index": "node index"}

#: Abscissae that are computed from the geometry rather than read from a field.
_DERIVED_AXES = {"arclength", "index"}


@MatplotLibPart.register()
class MatplotLibMainAxes(MatplotLibAxes):
    """The figure's own axes, driven as an ordinary x-y graph.

    It is a :py:class:`~pyoomph.output.plotting.MatplotLibAxes`, which is what lets every line part
    attach to it unchanged and reuses its range accumulation, its secondary y-axis and its legend.
    What differs is that it takes over the figure's ``gca()`` instead of adding a new axes on top.
    That matters beyond tidiness: ``gca()`` is the axes that ``perform_plot`` paints with
    ``background_color``, the one that ``plt.gca()``-based parts (polygons, tracers) draw into, and
    the one the bifurcation GUI identifies as ``figure.axes[0]``. An inset placed over an unused
    ``gca()`` would leave all three of them pointing at the wrong axes.

    Create it with :py:meth:`MatplotlibPlotter1D.set_axes`, or just read
    :py:attr:`MatplotlibPlotter1D.main_axes`.
    """
    mode = "mainaxes"
    #: Below the overlays (10) and below the lines, which take the axes' index plus one.
    zindex = 0.0
    #: Before the line parts, so that ``ax`` exists by the time the first of them asks for it.
    preprocess_order = -10.0

    #: Logarithmic x-axis.
    xlog:bool=False
    #: Logarithmic y-axis.
    ylog:bool=False
    #: Logarithmic secondary y-axis.
    y2log:bool=False
    #: Draw a grid at the major ticks.
    grid:bool=False
    #: Draw a fainter grid at the minor ticks as well.
    minor_grid:bool=False
    #: Fraction of the data range added as padding above and below an automatically determined
    #: y-range, so that a curve does not sit exactly on the frame.
    margin_y:float=0.05
    #: The same in x. Zero by default: an x-axis that runs exactly from the first to the last node
    #: is what one wants for a spatial profile.
    margin_x:float=0.0
    #: Key under which the axis ranges are persisted across output steps. Defaults to the title.
    range_key:str | None=None
    #: Formatter for the x tick labels, e.g. a format string like ``"{x:.2f}"`` or a matplotlib
    #: Formatter.
    xtick_format:Any=None
    #: The same for the y tick labels.
    ytick_format:Any=None
    #: The same for the secondary y tick labels.
    y2tick_format:Any=None

    def __init__(self,plotter:"MatplotlibPlotter"):
        super().__init__(plotter)
        self._uses_y2=False
        self._label_suggestions:dict[str,list[tuple[str,str]]]={"x":[],"y":[],"y2":[]}
        self._ranges:dict[str,Any]={}
        #: Unit string forced by set_axes(xunit=...), which also rescaled the data - so the unit a
        #: field reports for itself no longer applies and must not be appended as well.
        self._forced_units:dict[str,str | None]={"x":None,"y":None,"y2":None}

    def check_pos(self):
        # A main axes has no position to resolve - it IS the figure. The inherited implementation
        # would otherwise insist on a position= or xpos=/ypos= and raise.
        self.xpos,self.ypos=0.0,0.0

    def pre_process(self):
        self.check_pos()
        self.ax=plt.gcf().gca()

    def suggest_labels(self,x:tuple[str,str] | None=None,y:tuple[str,str] | None=None,use_y2:bool=False):
        """Offer axis labels, as (name, unit) pairs, derived from what is being plotted.

        Collected rather than assigned, because several parts can share an axis and the honest label
        then names all of them. An explicit xlabel/ylabel always wins over any suggestion, and the
        name and the unit are kept apart so that a set_axes(yunit=...) - which rescales the data -
        can replace the unit the field reports for itself.
        """
        if x is not None and x not in self._label_suggestions["x"]:
            self._label_suggestions["x"].append(x)
        key="y2" if use_y2 else "y"
        if y is not None and y not in self._label_suggestions[key]:
            self._label_suggestions[key].append(y)
        if use_y2:
            self._uses_y2=True

    def _resolve_label(self,which:str)->str | None:
        explicit={"x":self.xlabel,"y":self.ylabel,"y2":self.y2label}[which]
        forced=self._forced_units[which]
        if explicit is not None:
            if explicit=="" or not forced:
                return explicit  # including "" for a deliberately blank axis
            return explicit+" "+forced
        sugg=self._label_suggestions[which]
        if not sugg:
            return None
        names=[n for n,_u in sugg]
        units={u for _n,u in sugg if u}
        # Three is where a joined label stops being readable; beyond that the legend is the place to
        # say what is what.
        text=", ".join(names[:3])+(", ..." if len(names)>3 else "")
        if forced:
            return text+" "+forced
        # Only when every contributor agrees: two fields in different units share no single one.
        if len(units)==1 and len(names)>0:
            return text+" "+units.pop()
        return text

    def _range_object(self,which:str,mode:Any):
        key=(self.range_key or self.title or "mainaxes")+":"+which
        if key not in self._ranges:
            self._ranges[key]=self.plotter.get_range_object(key,mode=mode)
        return self._ranges[key]

    def consider_range(self,xmin:float | None=None,xmax:float | None=None,ymin:float | None=None,ymax:float | None=None,use_y2:bool=False):
        """Accumulate the data range, honouring the persistent range modes.

        ``"grow"`` and ``"firststep"`` are routed through the plotter's range objects, so they
        survive from one output step to the next and land in the ``_cb_ranges`` files beside the
        colorbar ranges. Everything else falls through to the inherited per-plot accumulation.
        """
        for which,lo,hi in (("x",xmin,xmax),("y2" if use_y2 else "y",ymin,ymax)):
            mode=getattr(self,"rangemode_"+which)
            if mode in ("auto","fixed"):
                continue
            if lo is None or hi is None:
                continue
            rng=self._range_object(which,("fixed" if mode=="firststep" else mode))
            rng.consider_range(lo,hi)
        super().consider_range(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,use_y2=use_y2)

    def _limits_for(self,which:str)->tuple[float | None,float | None]:
        mode=getattr(self,"rangemode_"+which)
        lo,hi={"x":(self.xmin,self.xmax),"y":(self.ymin,self.ymax),"y2":(self.y2min,self.y2max)}[which]
        if mode not in ("auto","fixed"):
            rng=self._range_object(which,("fixed" if mode=="firststep" else mode))
            if rng.vmin is not None:
                lo=rng.vmin
            if rng.vmax is not None:
                hi=rng.vmax
        if mode=="auto" and lo is not None and hi is not None:
            margin=self.margin_x if which=="x" else self.margin_y
            is_log={"x":self.xlog,"y":self.ylog,"y2":self.y2log}[which]
            if margin>0 and not is_log:
                # A relative padding is meaningless on a log axis; matplotlib pads that one itself.
                if hi>lo:
                    pad=margin*(hi-lo)
                    lo,hi=lo-pad,hi+pad
                else:
                    # A constant field would otherwise ask for a zero-height axis, which matplotlib
                    # renders as an arbitrary +-1e-7 window around it.
                    pad=max(abs(hi),1.0)*margin
                    lo,hi=lo-pad,hi+pad
        return lo,hi

    def _apply_ticks(self,axis:Any,fmt:Any):
        if fmt is None:
            return
        if isinstance(fmt,str):
            axis.set_major_formatter(matplotlib.ticker.StrMethodFormatter(fmt)) #type:ignore
        else:
            axis.set_major_formatter(fmt) #type:ignore

    def add_to_plot(self):
        if self.invisible:
            return
        assert self.ax is not None
        ax=self.ax
        l,b,r,t=cast("MatplotlibPlotter1D",self.plotter).margins
        ax.set_position([l,b,r-l,t-b]) #type:ignore
        if self.plotter.aspect_ratio:
            # Applied here rather than left to the inherited _reset_before_plot, which runs before
            # define_plot and would therefore miss a plotter that asks for it there - which is the
            # only place a plot definition ever sets it. Equal axes are what an (x,y) curve of a mesh
            # wants, so that a circular interface comes out round.
            ax.set_aspect(1 if self.plotter.aspect_ratio is True else self.plotter.aspect_ratio) #type:ignore
        if self.xlog:
            ax.set_xscale("log")
        if self.ylog:
            ax.set_yscale("log")
        if self.title:
            ax.set_title(self.title,size=self.textsize) #type:ignore
        xlab,ylab=self._resolve_label("x"),self._resolve_label("y")
        if xlab is not None:
            ax.set_xlabel(xlab,size=self.textsize) #type:ignore
        if ylab is not None:
            ax.set_ylabel(ylab,size=self.textsize,color=self.ylabel_color) #type:ignore
        y2lab=self._resolve_label("y2")
        if self._uses_y2 or y2lab is not None:
            # Created whenever anything actually uses the secondary axis. The inherited version only
            # creates the twin when y2label is set, so a line with use_y2=True and no label found
            # ax_y2 still None and crashed on plotting into it.
            if self.ax_y2 is None:
                self.ax_y2=ax.twinx()
            if y2lab is not None:
                self.ax_y2.set_ylabel(y2lab,size=self.textsize,color=self.y2label_color) #type:ignore
            if self.y2log:
                self.ax_y2.set_yscale("log")
        if self.ax_y2 is not None:
            # twinx() shares the x-axis - that part was always right - but it copies the parent's
            # RECTANGLE once, when it is created, and does not follow later set_position() calls. The
            # twin can be created before the margins above are applied (the inherited add_to_plot
            # makes one as soon as y2label is set), so it kept matplotlib's default subplot rect and
            # drew a second frame a few percent inside the real one: measured (0.125,0.11,0.775,0.77)
            # against the main axes' (0.1,0.13,0.8,0.82). Re-applying the position is the whole fix;
            # the shared x-axis needs nothing.
            self.ax_y2.set_position(ax.get_position()) #type:ignore
            # A twin draws its own frame on top of the parent's. Aligned they coincide exactly, but
            # only the right spine carries anything, and leaving the rest visible doubles every line
            # of the box at half opacity wherever antialiasing disagrees.
            for side in ("top","bottom","left"):
                self.ax_y2.spines[side].set_visible(False) #type:ignore
        # One bound at a time, unlike the inherited version, which applies a limit only when both
        # ends are known - so "ymin=0, autoscale the top" was not expressible at all.
        xlo,xhi=self._limits_for("x")
        if xlo is not None:
            ax.set_xlim(left=xlo)
        if xhi is not None:
            ax.set_xlim(right=xhi)
        ylo,yhi=self._limits_for("y")
        if ylo is not None:
            ax.set_ylim(bottom=ylo)
        if yhi is not None:
            ax.set_ylim(top=yhi)
        if self.ax_y2 is not None:
            y2lo,y2hi=self._limits_for("y2")
            if y2lo is not None:
                self.ax_y2.set_ylim(bottom=y2lo)
            if y2hi is not None:
                self.ax_y2.set_ylim(top=y2hi)
        if self.ticksize is not None:
            ax.tick_params(axis="both",which="major",labelsize=self.ticksize) #type:ignore
            if self.ax_y2 is not None:
                self.ax_y2.tick_params(axis="both",which="major",labelsize=self.ticksize) #type:ignore
        self._apply_ticks(ax.xaxis,self.xtick_format)
        self._apply_ticks(ax.yaxis,self.ytick_format)
        if self.ax_y2 is not None:
            self._apply_ticks(self.ax_y2.yaxis,self.y2tick_format)
        if self.grid:
            ax.grid(True,which="major",alpha=0.3) #type:ignore
        if self.minor_grid:
            ax.minorticks_on()
            ax.grid(True,which="minor",alpha=0.15) #type:ignore
        if self.hide_y_ticks:
            ax.tick_params(axis="y",which="both",left=False,right=False,labelleft=False) #type:ignore
        ax.patch.set_alpha(self.alpha) #type:ignore
        self._publish_view()

    def _publish_view(self):
        # The plotter's xmin..ymax are what MatplotLibOverlayBase._map_x maps a scale bar, an arrow
        # key or a scaled image through. On a graph they mean the axis range rather than a spatial
        # box; keeping them in step with the axes is what makes those overlays land where asked.
        assert self.ax is not None
        self.plotter.xmin,self.plotter.xmax=self.ax.get_xlim()
        self.plotter.ymin,self.plotter.ymax=self.ax.get_ylim()

    def post_process(self):
        super().post_process()
        if not self.invisible and self.ax is not None:
            self._publish_view()  # again, now that matplotlib has autoscaled whatever we left open


@MatplotLibPart.register()
class MatplotLibGraphLine(MatplotLibLinePlot):
    """A curve in an x-y graph: any nodal quantity of a 1d mesh against any other.

    It generalises :py:class:`~pyoomph.output.plotting.MatplotLibLinePlot` in exactly two ways. The
    abscissa is selectable instead of being hard-wired to the first coordinate, and the points are
    ordered by the mesh connectivity instead of by ascending x. The second is what makes a curve
    that folds back - a closed interface, a helix, the shape of a mesh in the plane - come out as
    the curve it is rather than as a zig-zag; it also keeps disconnected pieces of a domain apart
    instead of joining them with a chord.

    With a ``colorbar``, the curve is drawn as a colour-coded line collection instead of a plain
    line, in the same way an interface is coloured on a 2d plot.
    """
    mode = "graphline"

    #: What goes on the abscissa. See :py:meth:`MatplotlibPlotter1D.add_plot` for the accepted
    #: values. Never inferred from the nodal dimension of the mesh: ``coordinate_x`` is the abscissa
    #: for a 1d mesh in 1d, in 2d and in 3d space alike.
    xaxis:"str | NPFloatArray" = "coordinate_x"
    #: ``"connectivity"``, ``"sort_by_x"`` or ``"as_is"``.
    order:str = "connectivity"
    #: Off, in favour of :py:attr:`order`. The inherited argsort is only correct for a graph that is
    #: monotonic in its abscissa.
    sort_by_x = False
    #: Order the polylines among each other along a Cartesian direction, e.g. ``"x+"``.
    sort_along_axis:"SortAlongAxis | None" = None
    #: Order the polylines by their distance from this point, closest first.
    start_near_point:Any = None
    #: Added to every arclength, e.g. to measure it from somewhere other than the first node.
    arclength_offset:float = 0.0

    #: Colour the curve by a field instead of drawing it in a single colour.
    colorbar:"MatplotLibColorbar | None" = None
    #: The field doing the colouring. ``None`` colours by the plotted field itself.
    colorfield:str | None = None
    #: Outline the colour-coded curve in this colour, to lift it off a busy background.
    border_color:Any = None
    #: Width of that outline, added on both sides of the line.
    border_width:float = 3

    #: Draw a marker only at every n-th point.
    markevery:Any = None
    #: Opacity of the line.
    alpha:float | None = None

    def __init__(self,plotter:"MatplotlibPlotter"):
        super().__init__(plotter)
        self._ninter:int | None=None
        self._colordata:list[NPFloatArray]=[]
        self._lagrangian_coordinates:NPFloatArray | None=None

    # ---------------------------------------------------------------- data extraction

    def _mesh_segments(self)->list[list[int]]:
        """The node indices of each polyline of the mesh, in traversal order.

        ``get_interface_line_segments`` walks the element connectivity and keeps the interior node
        of a quadratic line element in its place (pyoomph stores a line3 as left, middle, right), so
        it yields a curve that is right even where no coordinate is monotonic along it. That is the
        whole reason for not sorting by x here.
        """
        cache=self.mshcache
        if len(cache.elem_indices)==0:
            return []  # the walk asserts on the intermediate-node count, which an empty mesh never sets
        if cache.discontinuous:
            raise RuntimeError("Cannot follow the mesh connectivity on a discontinuous mesh data cache")
        segs,self._ninter=cache.get_interface_line_segments()
        if self.sort_along_axis is not None or self.start_near_point is not None:
            segs=sort_line_segments(cache.get_coordinates(),segs,sort_along_axis=self.sort_along_axis,
                                    start_near_point=self.start_near_point,
                                    spatial_unit=self.plotter.get_problem(self.problem_name).get_scaling("spatial"),
                                    whom="add_plot(...)")
        return segs

    def _coordinate_array(self,lagrangian:bool)->NPFloatArray:
        if not lagrangian:
            assert self._coordinates is not None
            return self._coordinates
        if self._lagrangian_coordinates is None:
            self._lagrangian_coordinates=self.mshcache.get_coordinates(lagrangian=True)
        return self._lagrangian_coordinates

    def _resolve_axis_data(self,spec:"str | NPFloatArray",seg:Sequence[int])->NPFloatArray:
        """One polyline's worth of values for one axis.

        Resolved per polyline rather than once for all nodes, because the arclength restarts at every
        disconnected piece, and because a closed loop repeats its first node index at the end - in a
        single array indexed by node, that node's zero would be overwritten by the total length.
        """
        idx:NPIntArray=numpy.asarray(seg,dtype=numpy.int64) #type:ignore
        if isinstance(spec,numpy.ndarray):
            if len(spec)!=len(self.mshcache.nodal_values):
                raise ValueError("An array passed as an axis must have one entry per node of the mesh ("
                                 +str(len(self.mshcache.nodal_values))+"), but it has "+str(len(spec)))
            return spec[idx]
        if spec=="index":
            return numpy.arange(len(idx),dtype=numpy.float64) #type:ignore
        if spec=="arclength":
            pts=self._coordinate_array(False)[:,idx]
            steps:NPFloatArray=numpy.sqrt(numpy.sum(numpy.diff(pts,axis=1)**2,axis=0)) #type:ignore
            return numpy.concatenate([[0.0],numpy.cumsum(steps)])+self.arclength_offset #type:ignore
        if spec in _AXIS_PRETTY_NAMES and spec not in _DERIVED_AXES:
            lagrangian=spec.startswith("lagrangian")
            coords=self._coordinate_array(lagrangian)
            direction="xyz".index(spec[-1])
            if direction>=coords.shape[0]:
                raise RuntimeError("Cannot plot against '"+spec+"': this mesh only has "+str(coords.shape[0])
                                   +" coordinate direction(s). A LineMesh carries a y (or z) coordinate only "
                                   "when it is created with nodal_dimension=2 (or 3).")
            return coords[direction][idx]
        data=self.mshcache.get_data(spec)
        if data is None:
            raise RuntimeError("Cannot use '"+str(spec)+"' as an axis: it is neither a coordinate, nor "
                               "'arclength'/'index', nor a nodal field or local expression of this mesh. "
                               "Available fields: "+", ".join(self.mshcache.get_default_output_fields()))
        arr:NPFloatArray=numpy.asarray(data) #type:ignore
        if arr.ndim>1:
            # A vector field asked for as an axis: use its magnitude, as the 2d colour plots do.
            arr=numpy.sqrt(numpy.sum(arr**2,axis=0)) #type:ignore
        return arr[idx]

    def _resolve_segments(self)->list[tuple[NPFloatArray,NPFloatArray]]:
        if self._external_xdata is not None and self._external_ydata is not None:
            return [(self._external_xdata,self._external_ydata)]
        self._coordinates=self.mshcache.get_coordinates(lagrangian=self.use_lagrangian_coordinates)
        self._lagrangian_coordinates=None
        ydata=self.mshcache.get_data(self.field)
        if ydata is None:
            raise RuntimeError("Cannot plot the field "+str(self.field)+" of this mesh")
        self._data=numpy.asarray(ydata) #type:ignore
        if self.transform is not None:
            self._coordinates,self._data=self.transform.apply(self._coordinates,self._data)
        if self._data.ndim>1:
            self._data=numpy.sqrt(numpy.sum(self._data**2,axis=0)) #type:ignore

        if self.order=="connectivity":
            segs=self._mesh_segments()
        elif self.order in ("sort_by_x","as_is"):
            segs=[list(range(self._coordinates.shape[1]))]
        else:
            raise ValueError("Unknown order '"+str(self.order)+"', expected 'connectivity', 'sort_by_x' or 'as_is'")

        cdata=None
        if self.colorbar is not None:
            # A curve plots exactly one field, so the inherited str|list[str] is a str here.
            cfield=self.colorfield if self.colorfield is not None else cast(str,self.field)
            cdata=self._data if cfield==self.field else self._resolve_field_for_color(cfield)

        result:list[tuple[NPFloatArray,NPFloatArray]]=[]
        self._colordata=[]
        for seg in segs:
            x=self._resolve_axis_data(self.xaxis,seg)
            y=self._data[numpy.asarray(seg,dtype=numpy.int64)] #type:ignore
            if self.order=="sort_by_x":
                srt:NPIntArray=numpy.argsort(x) #type:ignore
                x,y=x[srt],y[srt]
                if cdata is not None:
                    self._colordata.append(numpy.asarray(cdata)[srt]) #type:ignore
            elif cdata is not None:
                self._colordata.append(numpy.asarray(cdata)[numpy.asarray(seg,dtype=numpy.int64)]) #type:ignore
            result.append((x,y))
        return result

    def _resolve_field_for_color(self,cfield:str)->NPFloatArray:
        data=self.mshcache.get_data(cfield)
        if data is None:
            raise RuntimeError("Cannot colour by '"+cfield+"': it is not a field of this mesh")
        arr:NPFloatArray=numpy.asarray(data) #type:ignore
        if arr.ndim>1:
            arr=numpy.sqrt(numpy.sum(arr**2,axis=0)) #type:ignore
        return arr

    # ---------------------------------------------------------------- labels and ranges

    def _label_for(self,spec:"str | list[str] | NPFloatArray | None")->tuple[str,str] | None:
        """The axis label for a quantity, as a (name, unit) pair."""
        if spec is None or isinstance(spec,numpy.ndarray):
            return None
        if isinstance(spec,(list,tuple)):
            return None  # a vector field plotted as its magnitude has no single natural name
        pretty=_AXIS_PRETTY_NAMES.get(spec,spec)
        try:
            if spec=="index":
                unit=""
            elif spec=="arclength":
                unit=self.mshcache.get_unit("coordinate_x",as_string=True,with_brackets=True)
            else:
                unit=self.mshcache.get_unit(spec,as_string=True,with_brackets=True)
        except Exception:
            # A local expression whose unit cannot be worked out should not cost us the whole plot.
            unit=""
        return (pretty,unit or "")

    def pre_process(self):
        super().pre_process()
        assert self.axes is not None
        if isinstance(self.axes,MatplotLibMainAxes):
            self.axes.suggest_labels(x=self._label_for(self.xaxis),y=self._label_for(self.field),
                                     use_y2=self.use_y2)
        if self.colorbar is not None and len(self._colordata)>0:
            allc:NPFloatArray=numpy.concatenate(self._colordata) #type:ignore
            scaled=allc*self.colorbar.factor+self.colorbar.offset
            self.colorbar.consider_range(float(numpy.amin(scaled)),float(numpy.amax(scaled))) #type:ignore

    # ---------------------------------------------------------------- drawing

    def _line_kwargs(self)->dict[str,Any]:
        kwargs=super()._line_kwargs()
        if self.alpha is not None:
            kwargs["alpha"]=self.alpha
        if self.markevery is not None:
            kwargs["markevery"]=self.markevery
        return kwargs

    def _draw_segment(self,ax:Any,x:NPFloatArray,y:NPFloatArray,kwargs:dict[str,Any],index:int)->list[Any]:
        if self.colorbar is None:
            return super()._draw_segment(ax,x,y,kwargs,index)
        if index>=len(self._colordata) or len(x)<2:
            return []
        d=self._colordata[index]*self.colorbar.factor+self.colorbar.offset
        points=numpy.array([x,y]).T.reshape(-1,1,2) #type:ignore
        segments=numpy.concatenate([points[:-1],points[1:]],axis=1) #type:ignore
        # Each drawn segment is coloured by the mean of its two end values; taking one end instead
        # would shift the whole colouring by half an element.
        values=0.5*(d[:-1]+d[1:])
        if self.border_color is not None and self.border_width>0:
            lc_b=collections.LineCollection(segments,colors=self.border_color, #type:ignore
                                            linewidths=self.linewidth+self.border_width,zorder=self.zindex-0.001)
            ax.add_collection(lc_b) #type:ignore
        lc=collections.LineCollection(segments,cmap=self.colorbar.cmap,norm=self.colorbar.get_norm(), #type:ignore
                                      linewidth=self.linewidth,zorder=self.zindex,alpha=self.alpha)
        lc.set_array(numpy.asarray(values)) #type:ignore
        ax.add_collection(lc) #type:ignore
        return []



@MatplotLibPart.register()
class MatplotLibGraphNodes(MatplotLibGraphLine):
    """The mesh nodes of a 1d domain, as markers on the graph.

    Useful for seeing what spatial adaptivity did, and for telling a coarse mesh from a smooth
    solution. Draw it on top of a :py:class:`MatplotLibGraphLine` of the same field.
    """
    mode = "graphnodes"
    linestyle = "none"
    markerstyle = "o"
    markersize = 4.0
    #: Skip the interior nodes of higher-order elements, leaving one marker per element corner.
    only_vertex_nodes:bool = False
    #: Node markers sit on a curve whose range has already been accounted for, so by default they do
    #: not widen the axes themselves.
    contributes_to_range:bool = False

    def _resolve_segments(self)->list[tuple[NPFloatArray,NPFloatArray]]:
        segs=super()._resolve_segments()
        if not self.only_vertex_nodes:
            return segs
        # ninter is the number of interior nodes per element, so every (ninter+1)-th point along a
        # polyline is an element end. It is set by the connectivity walk; without it there is nothing
        # to thin out.
        if self._ninter is None or self._ninter<=0:
            return segs
        step=self._ninter+1
        return [(x[::step],y[::step]) for x,y in segs]


@MatplotLibPart.register()
class MatplotLibGraphElementBorders(MatplotLibGraphLine):
    """The element boundaries of a 1d domain, as thin vertical lines across the graph.

    Only meaningful when the abscissa is a coordinate, since it is the element ends in *that*
    quantity that are drawn.
    """
    mode = "graphborders"
    color = "0.8"
    linewidth = 0.5
    linestyle = "-"
    #: Matplotlib drawing order of the lines themselves, kept separate from ``zindex``. ``zindex``
    #: only orders the plotter's parts among each other, and the line parts raise theirs above the
    #: axes' so that they are added after it exists - borders that did the same ended up drawn over
    #: the very curves they annotate.
    draw_zorder:float = 0.5

    def _border_positions(self)->list[float]:
        segs=self._mesh_segments()
        step=(self._ninter or 0)+1
        positions:list[float]=[]
        for seg in segs:
            ends=list(seg[::step])
            if seg and (len(seg)-1)%step!=0:
                ends.append(seg[-1])
            positions+=[float(v) for v in self._resolve_axis_data(self.xaxis,ends)]
        return positions

    def _resolve_segments(self)->list[tuple[NPFloatArray,NPFloatArray]]:
        # Nothing goes through the normal line path; add_to_plot draws the vertical lines itself.
        return []

    def pre_process(self):
        assert self.axes is not None
        if self.axes.ax is None:
            self.axes.pre_process()
        if self.zindex<=self.axes.zindex:
            self.zindex=self.axes.zindex+1
        self._coordinates=self.mshcache.get_coordinates(lagrangian=self.use_lagrangian_coordinates)
        self._lagrangian_coordinates=None
        self._segments=[]
        self._plotdata=None
        self._positions=self._border_positions()

    def add_to_plot(self):
        assert self.axes is not None
        ax=self.axes.ax_y2 if self.use_y2 else self.axes.ax
        assert ax is not None
        xfact=self.axes.xfactor*self.xfactor
        for pos in self._positions:
            ax.axvline(pos*xfact,color=self.color,linewidth=self.linewidth, #type:ignore
                       linestyle=self.linestyle,alpha=self.alpha,zorder=self.draw_zorder)


class MatplotlibPlotter1D(MatplotlibPlotter):
    """Plots one-dimensional domains as an ordinary x-y graph.

    Used exactly like :py:class:`~pyoomph.output.plotting.MatplotlibPlotter` - subclass it, implement
    :py:meth:`~pyoomph.output.plotting.BasePlotter.define_plot`, and assign an instance to the
    problem's ``plotter`` - but the figure's axes is a graph with visible ticks, labels and an
    optional legend instead of a spatial map::

        class MyPlotter(MatplotlibPlotter1D):
            def define_plot(self):
                self.set_axes(ymin=-1.3, ymax=1.9, grid=True, legend=True)
                self.add_plot("domain/u", color="navy", linewidth=2, label="$u$")
                self.add_time_label("top right")

    The abscissa is ``coordinate_x`` unless another one is asked for, in one dimension of space and
    in three alike. Everything else about a plot is inherited: file names and formats, eigenvector
    plotting, the merge of a mesh distributed over MPI ranks, and the overlays.

    Args:
        problem: The problem to plot.
        filetrunk: Trunk of the file name to save to, without extension.
        fileext: Extension (or list of extensions) to save.
        eigenvector: If set, plot this eigenvector instead of the solution.
        eigenmode: How to render an eigenvector (``"abs"``, ``"real"``, ``"imag"``).
        add_eigen_to_mesh_positions: Add the eigenvector of the mesh positions to the base positions.
        position_eigen_scale: Scale the eigenvector added to the mesh positions.
        eigenscale: Scale the whole eigenvector.
    """

    def __init__(self,problem:"Problem | None"=None,filetrunk:str="plot_{:05d}",fileext:str | list[str]="png",eigenvector:int | None=None,eigenmode:"MeshDataEigenModes"="abs",add_eigen_to_mesh_positions:bool=True,position_eigen_scale:float=1,eigenscale:float=1):
        super().__init__(problem,filetrunk=filetrunk,fileext=fileext,eigenvector=eigenvector,eigenmode=eigenmode,add_eigen_to_mesh_positions=add_eigen_to_mesh_positions,position_eigen_scale=position_eigen_scale,eigenscale=eigenscale)
        # A graph is not a spatial map: an equal aspect ratio and the fullscreen stripping of ticks
        # and margins are exactly what must not happen here. Turning the two flags off, rather than
        # overriding _reset_before_plot, makes the inherited figure setup collapse to a no-op and
        # keeps everything that reads them - notably the bifurcation GUI, which letterboxes a pane to
        # the aspect ratio a plotter asked for - agreeing with what is actually drawn.
        self.aspect_ratio=False
        self.fullscreen=False
        #: The graph rectangle within the figure, as (left, bottom, right, top) figure fractions.
        #: Wider defaults than matplotlib's, because an axis label with a unit needs the room.
        self.margins:tuple[float,float,float,float]=(0.12,0.13,0.97,0.93)
        self._main_axes:MatplotLibMainAxes | None=None

    # ---------------------------------------------------------------- the graph axes

    @property
    def main_axes(self)->MatplotLibMainAxes:
        """The graph itself, created on first access.

        Recreated for every plot, like every other part, so settings made in ``define_plot`` do not
        leak from one output step into the next.
        """
        if self._main_axes is None:
            res=self._add_part("mainaxes")
            assert isinstance(res,MatplotLibMainAxes)
            self._main_axes=res
        return self._main_axes

    def _default_axes(self)->MatplotLibAxes | None:
        return self.main_axes

    def _overlay_frame(self)->tuple[float,float,float,float]:
        return self.margins

    def set_axes(self,*,title:str | None=None,xlabel:str | None=None,ylabel:str | None=None,y2label:str | None=None,
                 xmin:ExpressionNumOrNone=None,xmax:ExpressionNumOrNone=None,ymin:ExpressionNumOrNone=None,ymax:ExpressionNumOrNone=None,
                 y2min:ExpressionNumOrNone=None,y2max:ExpressionNumOrNone=None,
                 xlog:bool | None=None,ylog:bool | None=None,y2log:bool | None=None,
                 rangemode_x:Any=None,rangemode_y:Any=None,rangemode_y2:Any=None,
                 grid:bool | None=None,minor_grid:bool | None=None,
                 legend:bool | None=None,legend_position:str | None=None,
                 ticksize:float | None=None,textsize:float | None=None,
                 margins:tuple[float,float,float,float] | None=None,
                 margin_x:float | None=None,margin_y:float | None=None,
                 xfactor:float | None=None,yfactor:float | None=None,
                 xunit:ExpressionNumOrNone=None,yunit:ExpressionNumOrNone=None,
                 ylabel_color:str | None=None,y2label_color:str | None=None,
                 hide_y_ticks:bool | None=None,
                 xtick_format:Any=None,ytick_format:Any=None,y2tick_format:Any=None,
                 range_key:str | None=None)->MatplotLibMainAxes:
        """
        Configures the graph. Every argument is optional; anything left out keeps its default.

        Args:
            title: Title above the graph. Also the key under which the axis ranges persist.
            xlabel: Label of the x-axis. ``None`` derives it from what is plotted, ``""`` leaves it blank.
            ylabel: Label of the y-axis, same convention.
            y2label: Label of the secondary y-axis, which is created as soon as anything uses it.
            xmin: Lower x-limit. Each limit is applied on its own, so half a range is fine.
            xmax: Upper x-limit.
            ymin: Lower y-limit.
            ymax: Upper y-limit.
            y2min: Lower limit of the secondary y-axis.
            y2max: Upper limit of the secondary y-axis.
            xlog: Logarithmic x-axis.
            ylog: Logarithmic y-axis.
            y2log: Logarithmic secondary y-axis.
            rangemode_x: How the x-range behaves over an output series, see below.
            rangemode_y: The same for y. ``"grow"`` is what a movie usually wants.
            rangemode_y2: The same for the secondary y-axis.
            grid: Draw a grid at the major ticks.
            minor_grid: Also draw one at the minor ticks.
            legend: ``True`` puts a legend at matplotlib's ``"best"`` position.
            legend_position: An explicit legend location, e.g. ``"upper right"``.
            ticksize: Font size of the tick labels.
            textsize: Font size of the axis labels and the title.
            margins: The graph rectangle as (left, bottom, right, top) figure fractions.
            margin_x: Padding added to an automatic x-range, as a fraction of it.
            margin_y: The same in y. Defaults to 0.05, so a curve does not touch the frame.
            xfactor: Factor all x-data is multiplied with.
            yfactor: Factor all y-data is multiplied with.
            xunit: Express the x-axis in this unit, e.g. ``milli*meter``. Sets ``xfactor`` and appends the unit to the label.
            yunit: The same for the y-axis.
            ylabel_color: Colour of the y-axis label.
            y2label_color: Colour of the secondary y-axis label.
            hide_y_ticks: Remove the y ticks and their labels entirely.
            xtick_format: Format string like ``"{x:.2f}"`` or a matplotlib Formatter for the x ticks.
            ytick_format: The same for the y ticks.
            y2tick_format: The same for the secondary y ticks.
            range_key: Key under which the ranges persist, if the title is not a good one.

        The range modes are ``"auto"`` (rescale to each output step), ``"fixed"`` (use the limits
        given here), ``"grow"`` (the union over all steps so far), ``"firststep"`` (lock onto the
        first step) and an explicit ``(lo,hi)`` pair.

        Returns:
            The graph, so that anything not covered here can be set on it directly.
        """
        ax=self.main_axes
        if legend and legend_position is None:
            legend_position="best"
        kwargs:dict[str,Any]={"title":title,"xlabel":xlabel,"ylabel":ylabel,"y2label":y2label,
                              "xlog":xlog,"ylog":ylog,"y2log":y2log,
                              "rangemode_x":rangemode_x,"rangemode_y":rangemode_y,"rangemode_y2":rangemode_y2,
                              "grid":grid,"minor_grid":minor_grid,"legend_position":legend_position,
                              "ticksize":ticksize,"textsize":textsize,
                              "margin_x":margin_x,"margin_y":margin_y,
                              "xfactor":xfactor,"yfactor":yfactor,
                              "ylabel_color":ylabel_color,"y2label_color":y2label_color,
                              "hide_y_ticks":hide_y_ticks,"xtick_format":xtick_format,
                              "ytick_format":ytick_format,"y2tick_format":y2tick_format,
                              "range_key":range_key}
        ax.set_kwargs(kwargs)
        if margins is not None:
            self.margins=margins
        # The unit sets the factor and decorates the label, exactly as a colorbar's unit does.
        for unit,axname in ((xunit,"x"),(yunit,"y")):
            if unit is None:
                continue
            ustr,_num,factor=unit_to_string(unit,estimate_prefix=True)
            setattr(ax,axname+"factor",getattr(ax,axname+"factor")*factor)
            if ustr!="":
                ax._forced_units[axname]="["+ustr+"]"
        # Limits carry units of their own, and a limit that is given fixes that end of the range.
        for name,value in (("xmin",xmin),("xmax",xmax),("ymin",ymin),("ymax",ymax),("y2min",y2min),("y2max",y2max)):
            if value is not None:
                setattr(ax,name,self.ensure_spatial_nondim(value))
        for which,lo,hi,given in (("x",xmin,xmax,rangemode_x),("y",ymin,ymax,rangemode_y),("y2",y2min,y2max,rangemode_y2)):
            if given is None and lo is not None and hi is not None:
                setattr(ax,"rangemode_"+which,"fixed")
        return ax

    def set_view(self,xmin:ExpressionNumOrNone=None,ymin:ExpressionNumOrNone=None,xmax:ExpressionNumOrNone=None,ymax:ExpressionNumOrNone=None,center:list[ExpressionOrNum] | None=None,size:list[ExpressionOrNum] | None=None):
        """
        Sets the axis range of the graph. On a graph the "view" is the range of the two axes, not a
        spatial box, but the method keeps the inherited name on purpose: the bifurcation GUI locks a
        zoomed-in view by replacing ``set_view`` on the plotter, and a differently named method would
        leave a 1d pane unable to hold a zoom.

        Args:
            xmin: Left end of the x-axis.
            ymin: Bottom of the y-axis.
            xmax: Right end of the x-axis.
            ymax: Top of the y-axis.
            center: Centre of the view, as an alternative to the four limits.
            size: Width and height of the view, to be used with ``center``.
        """
        if center is not None and size is not None:
            self.set_view(xmin=center[0]-size[0]/2,xmax=center[0]+size[0]/2,ymin=center[1]-size[1]/2,ymax=center[1]+size[1]/2)
            return
        ax=self.main_axes
        if xmin is not None:
            ax.xmin=self.xmin=self.ensure_spatial_nondim(xmin)
        if xmax is not None:
            ax.xmax=self.xmax=self.ensure_spatial_nondim(xmax)
        if ymin is not None:
            ax.ymin=self.ymin=self.ensure_spatial_nondim(ymin)
        if ymax is not None:
            ax.ymax=self.ymax=self.ensure_spatial_nondim(ymax)
        if ax.xmin is not None and ax.xmax is not None:
            ax.rangemode_x="fixed"
        if ax.ymin is not None and ax.ymax is not None:
            ax.rangemode_y="fixed"

    def _reset_before_plot(self):
        super()._reset_before_plot()
        self._main_axes=None  # the axes is a part like any other, and the parts were just cleared

    def perform_plot(self):
        if not self._embedded:
            # The inherited code sizes the figure only inside set_view's equal-aspect branch, which a
            # graph never takes, so image_size and dpi would otherwise do nothing at all here. Done
            # here rather than in _reset_before_plot because define_plot runs in between, and setting
            # image_size there is the normal way to ask for a figure size. Left alone while embedded:
            # there the host canvas owns the figure size.
            plt.gcf().set_size_inches(self.image_size[0]/self.dpi,self.image_size[1]/self.dpi)
        super().perform_plot()

    # ---------------------------------------------------------------- adding plots

    def _resolve_1d_mode(self,infield:str,mode:str | None,problem_name:str)->str:
        if mode is not None:
            return mode  # an explicit mode is honoured, so "image", "lineplot" and so on still work
        parts=infield.split("/")
        problem=self.get_problem(problem_name=problem_name)
        if problem.get_mesh(infield,return_None_if_not_found=True) is not None:
            field,mshname=None,infield
        elif len(parts)<2:
            field,mshname=None,infield
        else:
            field,mshname="/".join(parts[-1:]),"/".join(parts[0:-1])
        msh=problem.get_mesh(mshname,return_None_if_not_found=True)
        if msh is None:
            raise ValueError("Cannot find the mesh "+mshname+" in the problem to plot "+str(field))
        dim=msh.get_dimension()
        if dim!=1:
            raise RuntimeError("MatplotlibPlotter1D plots one-dimensional domains, but '"+mshname+"' is "
                               +str(dim)+"-dimensional. Use MatplotlibPlotter for a spatial plot of it, or "
                               "name one of its interfaces, which is one dimension lower.")
        if field is not None:
            if msh.get_tracers(field,error_on_missing=False) is not None:
                raise RuntimeError("'"+field+"' is a tracer collection. Tracers are scattered over a spatial "
                                   "map and have no place on an x-y graph.")
            cached=self._get_mesh_data(msh,problem_name=problem_name,ignore_eigenfactors=True)
            if field not in cached.nodal_field_inds and field in cached.elemental_field_inds:
                raise RuntimeError("'"+field+"' is an elemental (D0/DL) field. Those are not delivered by the "
                                   "mesh data cache and cannot be plotted yet.")
        else:
            cached=self._get_mesh_data(msh,problem_name=problem_name,ignore_eigenfactors=True)
            if "coordinate_y" not in cached.nodal_field_inds:
                raise RuntimeError("Plotting the bare domain '"+mshname+"' draws the curve the mesh itself "
                                   "traces in the plane, which needs a y coordinate. This mesh has none - name "
                                   "a field to plot instead, or build the mesh with nodal_dimension=2.")
        return "graphline"

    def add_plot(self,infield:str,mode:str | None=None,transform:list[PlotTransform | None] | list[str | None] | str | PlotTransform | None=None,*,
                 xaxis:"str | NPFloatArray | None"=None,
                 order:str | None=None,sort_along_axis:"SortAlongAxis | None"=None,start_near_point:Any=None,
                 arclength_offset:float | None=None,
                 color:str | None=None,linewidth:float | None=None,linestyle:str | None=None,alpha:float | None=None,
                 marker:str | None=None,markersize:float | None=None,markevery:Any=None,label:str | None=None,
                 colorbar:MatplotLibColorbar | None=None,colorfield:str | None=None,
                 border_color:Any=None,border_width:float | None=None,
                 xfactor:float | None=None,yfactor:float | None=None,
                 axes:MatplotLibAxes | None=None,use_y2:bool | None=None,zindex:float | None=None,
                 problem_name:str="",**kwargs:Any)->Any:
        """
        Adds a curve of ``infield`` (e.g. ``"domain/u"``) to the graph.

        Without a ``mode`` the field is drawn against ``xaxis``, which defaults to ``coordinate_x``
        whether the 1d domain lives in one, two or three dimensions of space. Naming a bare domain
        instead of a field (e.g. ``"domain"``) draws the curve the mesh itself traces in the plane.
        With a ``colorbar``, the curve is colour-coded rather than drawn in a single colour.

        The points follow the mesh connectivity, so a curve that folds back on itself or closes into
        a loop comes out correctly, and a domain made of disconnected pieces is not joined up.

        Args:
            infield: Domain and field, e.g. ``"domain/u"``, or just a domain for its own curve.
            mode: Plotting mode. Inferred when left out.
            transform: A transform, or a list of them to draw the same data several times.
            xaxis: What goes on the abscissa. ``"coordinate_x"`` (the default), ``"coordinate_y"``,
                ``"coordinate_z"``, the ``"lagrangian_*"`` equivalents, ``"arclength"`` for the
                distance along the curve, ``"index"`` for the position along it, the name of any
                nodal field or local expression, or an array with one entry per node.
            order: ``"connectivity"`` (the default), ``"sort_by_x"`` or ``"as_is"``.
            sort_along_axis: Order disconnected pieces along a direction, e.g. ``"x+"``.
            start_near_point: Order them by distance from this point instead.
            arclength_offset: Added to every arclength.
            color: Line colour.
            linewidth: Line width.
            linestyle: Line style, e.g. ``"dashed"``.
            alpha: Opacity.
            marker: Marker style, e.g. ``"o"``.
            markersize: Marker size.
            markevery: Draw a marker only at every n-th point.
            label: Legend entry. Needs a legend on the axes, see :py:meth:`set_axes`.
            colorbar: Colour the curve by a field, using this colorbar for the scale.
            colorfield: The field doing the colouring. Defaults to the plotted field itself.
            border_color: Outline the colour-coded curve in this colour.
            border_width: Width of that outline.
            xfactor: Factor the x-data of this curve is multiplied with.
            yfactor: Factor the y-data of this curve is multiplied with.
            axes: Draw into an inset from :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_axes`
                instead of the main graph.
            use_y2: Draw against the secondary y-axis.
            zindex: Drawing order; higher is drawn on top.
            problem_name: Name of the problem, when several are plotted together.

        Returns:
            The added curve, or a list of them if several transforms were given.
        """
        resolved=self._resolve_1d_mode(infield,mode,problem_name)
        extra:dict[str,Any]={"xaxis":xaxis,"order":order,"sort_along_axis":sort_along_axis,
                             "start_near_point":start_near_point,"arclength_offset":arclength_offset,
                             "color":color,"linewidth":linewidth,"linestyle":linestyle,"alpha":alpha,
                             "marker":marker,"markersize":markersize,"markevery":markevery,"label":label,
                             "colorfield":colorfield,"border_color":border_color,"border_width":border_width,
                             "xfactor":xfactor,"yfactor":yfactor,"use_y2":use_y2,"zindex":zindex}
        extra.update(kwargs)
        # marker is the matplotlib spelling; the part inherits the attribute name markerstyle.
        if extra.pop("marker",None) is not None:
            extra["markerstyle"]=marker
        # A bare domain name means the curve the mesh traces, i.e. y against x.
        if resolved=="graphline" and self.get_problem(problem_name=problem_name).get_mesh(infield,return_None_if_not_found=True) is not None:
            infield=infield+"/coordinate_y"
        res=super().add_plot(infield,mode=resolved,transform=transform,colorbar=colorbar,
                             axes=axes if axes is not None else self.main_axes,problem_name=problem_name)
        for part in (res if isinstance(res,list) else [res]):
            part.set_kwargs(extra)
        return res

    def add_curve(self,mesh:str,*,xaxis:str="coordinate_x",yaxis:str="coordinate_y",colorbar:MatplotLibColorbar | None=None,colorfield:str | None=None,**kwargs:Any)->Any:
        """
        Draws the curve a 1d mesh traces in space, i.e. one coordinate against another.

        Args:
            mesh: The domain, e.g. ``"liquid/liquid_gas"`` for an interface of a 2d problem.
            xaxis: The coordinate on the abscissa.
            yaxis: The coordinate on the ordinate.
            colorbar: Colour the curve by a field, using this colorbar for the scale.
            colorfield: The field doing the colouring. Required together with ``colorbar`` here,
                since the plotted quantity is a coordinate.
            **kwargs: Anything :py:meth:`add_plot` accepts.

        Returns:
            The added curve.
        """
        if colorbar is not None and colorfield is None:
            raise ValueError("add_curve(colorbar=...) also needs colorfield=..., naming the field that "
                             "should colour the curve - colouring it by its own y coordinate is almost "
                             "never what is meant.")
        return self.add_plot(mesh+"/"+yaxis,xaxis=xaxis,colorbar=colorbar,colorfield=colorfield,**kwargs)

    def add_nodes(self,infield:str,*,only_vertex_nodes:bool=False,**kwargs:Any)->Any:
        """
        Marks the mesh nodes of a 1d domain on the graph.

        Args:
            infield: Domain and field, as for :py:meth:`add_plot`.
            only_vertex_nodes: Skip the interior nodes of higher-order elements.
            **kwargs: Anything :py:meth:`add_plot` accepts.

        Returns:
            The added markers.
        """
        return self.add_plot(infield,mode="graphnodes",only_vertex_nodes=only_vertex_nodes,**kwargs)

    def add_element_borders(self,mesh:str,*,xaxis:str="coordinate_x",**kwargs:Any)->Any:
        """
        Draws a thin vertical line at every element boundary of a 1d domain.

        Args:
            mesh: The domain.
            xaxis: The abscissa the element ends are taken in. Must match the one of the curves.
            **kwargs: Anything :py:meth:`add_plot` accepts.

        Returns:
            The added borders.
        """
        return self.add_plot(mesh+"/coordinate_x",mode="graphborders",xaxis=xaxis,**kwargs)

    def add_analytical(self,func:Callable[[NPFloatArray],NPFloatArray],*,xrange:tuple[float,float] | None=None,npoints:int=200,**kwargs:Any)->Any:
        """
        Draws an analytical curve, evaluated on a uniform grid, e.g. to compare against the solution.

        Args:
            func: Called with an array of abscissa values, returns the ordinate values.
            xrange: The interval to evaluate on. Defaults to the x-range of the graph, which then has
                to be known already - either set with :py:meth:`set_axes` or contributed by a curve
                added before this one.
            npoints: Number of points to evaluate at.
            **kwargs: Anything :py:meth:`~pyoomph.output.plotting.MatplotlibPlotter.add_external_data` accepts.

        Returns:
            The added curve.
        """
        if xrange is None:
            ax=self.main_axes
            if ax.xmin is None or ax.xmax is None:
                raise RuntimeError("add_analytical needs an x-range: pass xrange=(lo,hi), or set the "
                                   "x-limits with set_axes(xmin=..., xmax=...) first.")
            xrange=(ax.xmin,ax.xmax)
        xs:NPFloatArray=numpy.linspace(xrange[0],xrange[1],npoints) #type:ignore
        return self.add_external_data(xs,numpy.asarray(func(xs),dtype=numpy.float64),**kwargs) #type:ignore


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
