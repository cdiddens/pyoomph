"""Generate ``structure.svg``/``structure.pdf``, the overview figure of the design philosophy
chapter. This is not a tutorial example - it is the source of a committed figure, kept here so
that the picture can be regenerated when the chapter changes. It needs matplotlib and the gmsh
python module, neither of which the documentation build itself requires.

Left panel: an axisymmetric droplet on a substrate, surrounded by a gas dome, i.e. two separate
meshes that share the free surface. Right panel: the equation tree merged onto that geometry.

The mesh comes out of gmsh - the same mesher a :class:`GmshTemplate` drives - so that the figure
shows a real mesh rather than a drawing of one.
"""

import math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import gmsh

OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("structure"))

# ----------------------------------------------------------------------------- geometry & mesh

R = 1.0                       # base radius
THETA = math.radians(55)      # contact angle
RG = 2.0                      # radius of the gas dome
RS = R / math.sin(THETA)      # radius of the spherical cap
YC = -R / math.tan(THETA)     # center of the spherical cap
H = RS + YC                   # apex height


def build_mesh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    geo = gmsh.model.geo

    res_cl, res_apex, res_far = 0.05, 0.085, 0.26
    p_origin = geo.addPoint(0, 0, 0, res_cl * 1.6)
    p_cl = geo.addPoint(R, 0, 0, res_cl)
    p_apex = geo.addPoint(0, H, 0, res_apex)
    p_center = geo.addPoint(0, YC, 0, 1.0)
    p_gas_r = geo.addPoint(RG, 0, 0, res_far)
    p_gas_t = geo.addPoint(0, RG, 0, res_far)

    l_sub_drop = geo.addLine(p_origin, p_cl)
    l_iface = geo.addCircleArc(p_cl, p_center, p_apex)
    l_axis_drop = geo.addLine(p_apex, p_origin)
    l_sub_gas = geo.addLine(p_cl, p_gas_r)
    l_inf = geo.addCircleArc(p_gas_r, p_origin, p_gas_t)
    l_axis_gas = geo.addLine(p_gas_t, p_apex)

    s_drop = geo.addPlaneSurface([geo.addCurveLoop([l_sub_drop, l_iface, l_axis_drop])])
    s_gas = geo.addPlaneSurface([geo.addCurveLoop([l_sub_gas, l_inf, l_axis_gas, -l_iface])])
    geo.synchronize()
    gmsh.model.mesh.generate(2)

    out = {}
    for name, surf in (("droplet", s_drop), ("gas", s_gas)):
        tags, coords, _ = gmsh.model.mesh.getNodes(2, surf, includeBoundary=True)
        index = {t: i for i, t in enumerate(tags)}
        xy = coords.reshape(-1, 3)[:, :2]
        _, _, conn = gmsh.model.mesh.getElements(2, surf)
        tris = np.array([index[t] for t in conn[0]]).reshape(-1, 3)
        out[name] = Triangulation(xy[:, 0], xy[:, 1], tris)
    gmsh.finalize()
    return out


# ----------------------------------------------------------------------------- style

C_DROP = "#7fb3d5"
C_GAS = "#ddd6c8"
C_IFACE = "#c0392b"
C_SUB = "#7b5e3b"
C_AXIS = "#7f8c8d"
C_INF = "#2e86c1"
MONO = {"family": "DejaVu Sans Mono"}

fig = plt.figure(figsize=(9.9, 4.7))
axg = fig.add_axes([0.003, 0.015, 0.555, 0.970])
axt = fig.add_axes([0.578, 0.015, 0.418, 0.970])
for ax in (axg, axt):
    ax.set_axis_off()

# ----------------------------------------------------------------------------- left: geometry

tri = build_mesh()
axg.set_aspect("equal")
one = matplotlib.colors.ListedColormap
axg.tripcolor(tri["gas"], np.zeros(len(tri["gas"].x)), cmap=one([C_GAS]))
axg.triplot(tri["gas"], color="#928a7a", lw=0.4)
axg.tripcolor(tri["droplet"], np.zeros(len(tri["droplet"].x)), cmap=one([C_DROP]))
axg.triplot(tri["droplet"], color="#2c5f80", lw=0.45)

phi = np.linspace(0, math.pi / 2, 200)
axg.plot(RG * np.cos(phi), RG * np.sin(phi), color=C_INF, lw=2.2, solid_capstyle="round")
axg.plot([0, RG], [0, 0], color=C_SUB, lw=2.8, solid_capstyle="butt")
axg.plot([0, 0], [0, RG], color=C_AXIS, lw=1.4, ls=(0, (6, 4)))
ang = np.linspace(math.atan2(-YC, R), math.pi / 2, 200)
axg.plot(RS * np.cos(ang), YC + RS * np.sin(ang), color=C_IFACE, lw=3.0, solid_capstyle="round")
axg.plot([R], [0], "o", color=C_IFACE, ms=6.5, zorder=5)

axg.text(0.40, 0.20, '"droplet"', color="#0e344b", fontweight="bold", fontsize=11.5,
         ha="center", va="center", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
axg.text(1.05, 1.15, '"gas"', color="#4a4335", fontweight="bold", fontsize=11.5,
         ha="center", va="center", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

BOX = dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.8)  # keep labels readable on top of the mesh
lead = dict(fontsize=9.0, color="#333333", **MONO)
arrow = lambda c: dict(arrowstyle="-", color=c, lw=0.8, shrinkA=1, shrinkB=1)
axg.annotate('"droplet_gas"', xy=(0.72, 0.40), xytext=(1.05, 0.72), color=C_IFACE,
             arrowprops=arrow(C_IFACE), fontsize=9.4, bbox=BOX, **MONO)
axg.annotate('"gas_infinity"', xy=(RG * math.cos(0.78), RG * math.sin(0.78)), xytext=(1.30, 1.85),
             color=C_INF, arrowprops=arrow(C_INF), fontsize=9.4, bbox=BOX, **MONO)
axg.annotate('"droplet_substrate"', xy=(0.55, 0.0), xytext=(0.20, -0.40),
             ha="center", va="top", arrowprops=arrow(C_SUB), **lead)
axg.annotate('"gas_substrate"', xy=(1.55, 0.0), xytext=(1.62, -0.40),
             ha="center", va="top", arrowprops=arrow(C_SUB), **lead)
axg.annotate('"droplet_axisymm"', xy=(0.0, 0.26), xytext=(-0.62, 0.62),
             ha="center", va="bottom", arrowprops=arrow(C_AXIS), **lead)
axg.text(-0.06, 1.35, '"gas_axisymm"', ha="right", va="center", rotation=90, **lead)
axg.annotate("contact line", xy=(R, 0), xytext=(1.28, 0.30), fontsize=9.2, color="#333333",
             bbox=BOX, arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8, shrinkB=4))

axg.set_xlim(-1.28, 2.15)
axg.set_ylim(-0.58, 2.10)

# ----------------------------------------------------------------------------- right: tree

axt.set_xlim(0, 1)
axt.set_ylim(0, 1)

ELLIPSIS = object()  # placeholder row for what is left out to keep the example short

TREE = [
    (0, 'Problem', "#000000", True, None),
    (1, '"droplet"', "#0e344b", True, C_DROP),
    (2, 'PseudoElasticMesh()', None, False, None),
    (2, 'NavierStokesEquations(rho,mu)', None, False, None),
    (2, 'MeshFileOutput()', None, False, None),
    (2, '@"droplet_gas"', C_IFACE, True, None),
    (3, 'NavierStokesFreeSurface(sigma)', None, False, None),
    (3, 'ConnectMeshAtInterface()', None, False, None),
    (3, '@"droplet_substrate"', C_IFACE, True, None),
    (4, 'NavierStokesContactAngle(theta)', None, False, None),
    (2, ELLIPSIS, None, False, None),
    (1, '"gas"', "#4a4335", True, C_GAS),
    (2, 'PseudoElasticMesh()', None, False, None),
    (2, 'AdvectionDiffusionEquations("c_vap",D)', None, False, None),
    (2, '@"droplet_gas"', C_IFACE, True, None),
    (3, 'DirichletBC(c_vap=c_sat)', None, False, None),
    (2, ELLIPSIS, None, False, None),
]

DX, DY, Y0 = 0.055, 0.0505, 0.930
rows = [(0.030 + d * DX, Y0 - i * DY, d, lab, col, bold, sw)
        for i, (d, lab, col, bold, sw) in enumerate(TREE)]

for i, row in enumerate(rows):
    x, yy, depth = row[0], row[1], row[2]
    if depth == 0:
        continue
    j = i - 1
    while rows[j][2] >= depth:
        j -= 1
    px = x - DX + 0.011
    axt.plot([px, x - 0.013], [yy, yy], color="#b4b4b4", lw=0.8, zorder=1)
    axt.plot([px, px], [rows[j][1] - 0.011, yy], color="#b4b4b4", lw=0.8, zorder=1)

for x, yy, depth, label, color, bold, swatch in rows:
    if swatch is not None:
        axt.add_patch(plt.Rectangle((x - 0.010, yy - 0.010), 0.020, 0.020, facecolor=swatch,
                                    edgecolor="#666666", lw=0.5, zorder=3))
        x += 0.019
    if label is ELLIPSIS:  # visible enough to be read as "this continues", without a caption
        axt.text(x, yy, "...", va="center", ha="left", fontsize=13, zorder=3,
                 color="#5c5c5c", fontweight="bold", **MONO)
        axt.text(x + 0.090, yy, "more equations and interfaces", va="center", ha="left",
                 fontsize=10.2, zorder=3, color="#6f6f6f", style="italic")
        continue
    axt.text(x, yy, label, va="center", ha="left", fontsize=10.2, zorder=3,
             color=color or "#222222", fontweight="bold" if bold else "normal", **MONO)

for ext, kw in ((".pdf", {}), (".svg", {})):
    fig.savefig(OUT + ext, **kw)
print("written", OUT)
