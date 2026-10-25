"""Two plotting details that are invisible until a movie is made of a moving mesh.

Both are pure-Python parts of pyoomph/output/plotting.py, so they are tested against stub data
rather than against a solved problem: what is at stake is the bookkeeping, not the physics.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy
import pytest

from pyoomph.output.plotting import MatplotLibPersistentRange, MatplotLibTracers


# ----------------------------------------------------------------------------------------------
# A pinned colorbar/arrow-key range
# ----------------------------------------------------------------------------------------------

def test_a_pinned_range_holds_from_the_moment_it_is_created():
    """A range object is created lazily, inside the first pre_process - i.e. after the first output
    step's reset_range() has already run - and consider_range() ignores a pinned pair on purpose.
    So unless the constructor applies the pair, the first output step has no range at all and every
    part that reads one draws nothing: frame 0 of a movie comes out empty."""
    rng = MatplotLibPersistentRange(None, None, (2.0, 5.0))
    assert (rng.vmin, rng.vmax) == (2.0, 5.0)
    # And it stays pinned when data arrives that would widen any other mode.
    rng.consider_range(-100.0, 100.0)
    assert (rng.vmin, rng.vmax) == (2.0, 5.0)


@pytest.mark.parametrize("mode", ["current", "fixed", "grow"])
def test_the_other_modes_still_start_without_a_range(mode):
    """The counterpart: the three data-driven modes must keep taking their range from the data, so
    the fix above may not hand them one."""
    rng = MatplotLibPersistentRange(None, None, mode)
    assert (rng.vmin, rng.vmax) == (None, None)
    rng.consider_range(1.0, 3.0)
    assert (rng.vmin, rng.vmax) == (1.0, 3.0)


# ----------------------------------------------------------------------------------------------
# Clipping a trail to the domain it was drawn in
# ----------------------------------------------------------------------------------------------

class _StubCollection:
    """The handful of methods _add_trails asks a TracerCollection for."""

    def __init__(self, history):
        self._history = history

    def get_periodic_wraps(self):
        return []

    def get_ids(self):
        return list(self._history.keys())

    def get_dead_ids(self):
        return []

    def get_history(self, tid):
        return self._history[int(tid)]


def _segments_of(part, col):
    plt.figure()
    try:
        part._add_trails(col, 1.0)
        collections = plt.gca().collections
        if not collections:
            return []
        return [numpy.asarray(s) for s in collections[-1].get_segments()] #type:ignore
    finally:
        plt.close("all")


def _straight_trail():
    # Five samples walking up the y axis, from y=0.1 to y=0.9.
    return {1: numpy.array([[float(k), 0.5, 0.1 + 0.2 * k] for k in range(5)])}


def test_a_trail_is_drawn_whole_when_it_is_not_clipped():
    """The baseline, so that the clipped case below cannot pass by drawing nothing."""
    part = MatplotLibTracers(plotter=None) #type:ignore
    part.set_tracer_data("tracers", mesh=None, transform=None) #type:ignore
    part.trail = True
    assert len(_segments_of(part, _StubCollection(_straight_trail()))) == 4


def test_trail_clip_to_domain_drops_the_samples_that_left_the_domain():
    """A history sample is stored in the lab frame while the mesh moves, so an evaporating drop
    leaves its tracers' trails hanging in space it has since vacated. With the clip on, a sample
    outside the current domain is dropped and the trail is cut there rather than drawn through the
    gas - and both ends of a segment have to be inside, so the cut lands on the boundary."""
    part = MatplotLibTracers(plotter=None) #type:ignore
    part.set_tracer_data("tracers", mesh=None, transform=None) #type:ignore
    part.trail = True
    part.trail_clip_to_domain = True
    # Stands in for the triangulated domain: everything above y=0.55 has evaporated away.
    part._domain_clip_finder = lambda: (lambda _x, y: numpy.where(numpy.asarray(y) <= 0.55, 0, -1)) #type:ignore
    segs = _segments_of(part, _StubCollection(_straight_trail()))
    assert len(segs) == 2, "samples at y=0.7 and y=0.9 are outside and must not be joined"
    assert max(float(numpy.max(s[:, 1])) for s in segs) <= 0.55


def test_trail_clip_to_domain_is_off_by_default():
    """It costs a point location per history sample and a trifinder per frame, and on a fixed mesh
    it can only ever be a no-op."""
    assert MatplotLibTracers.trail_clip_to_domain is False
