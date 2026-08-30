#!/usr/bin/env python3
#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
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

"""UNDER-request detector: the reverse of scan_required_shapes.py, and per FUNCTION.

For every generated function it takes the required-shapes struct that the function itself passes to
fill_shape_buffer_for_point (or, for the non-integrated evaluators, the struct the C++ side fills
with, named after them) and checks that every shape buffer the function DEREFERENCES is flagged in
exactly that struct.

    python3 dev_docs/scripts/scan_required_shapes_reverse.py FILELIST [ROOT]

FILELIST is a file of .c paths, one per line, relative to ROOT (default ~/code/pyoomph_runs); absolute
paths work too.

Two things make this different from a naive scan, and both were needed to find a real defect:

  * PER FUNCTION. A whole-file union of the flags hides an under-request in one function behind
    another function's flags - which is precisely how shapes_required_Hessian could carry `psi` alone
    for a lid-driven cavity whose HessianVectorProduct body dereferences dx_shapes, while the ResJac
    struct next to it flagged dx_psi.
  * DEREFERENCE, not mention. `DX_SHAPE_FUNCTION_DECL(dX_testfunction) = shapeinfo->dX_shapes[...];`
    is emitted for every space loop; taking the address of an unfilled buffer is harmless. The alias
    is resolved and only its indexed uses inside the enclosing block count.

Unlike the over-request direction, a finding here IS a defect: the code reads a buffer that nothing
guarantees was filled. It only stays invisible while the runtime fill is coarser than the flags.
"""

import os,re,sys,collections
ROOT=os.path.expanduser(sys.argv[2] if len(sys.argv)>2 else "~/code/pyoomph_runs")
files=[l.strip() for l in open(sys.argv[1]) if l.strip()]
SET_RE=re.compile(r"functable->shapes_required_(\w+(?:\[\d+\])?)\.((?:\w+_shapes->)*)"
                  r"(?:continuous_spaces\[(\w+)\]\.)?([A-Za-z_0-9.]+)\s*=\s*true\s*;")
USE_RE=re.compile(r"my_func_table->shapes_required_(\w+(?:\[\d+\])?)")
READ_RE=re.compile(r"((?:\w*shapeinfo(?:->\w*shapeinfo)*))->(\w+)((?:\[[^\]\[]*\])*)")
IDX=re.compile(r"\[([^\]\[]*)\]")
CH={"":0,"bulk_shapes->":1,"opposite_shapes->":1,"bulk_shapes->bulk_shapes->":2,"opposite_shapes->bulk_shapes->":2}
BUF2FLAG={"shapes":"psi","dx_shapes":"dx_psi","dX_shapes":"dX_psi","dS_shapes":"dX_psi",
          "d2x_shapes":"d2x_psi","d2S_shapes":"d2x_psi",
          "shape_Pos":"Pos.psi","dx_shape_Pos":"Pos.dx_psi","dX_shape_Pos":"Pos.dX_psi","dS_shape_Pos":"Pos.dX_psi",
          "d2x_shape_Pos":"Pos.d2x_psi","d2S_shape_Pos":"Pos.d2x_psi",
          "shape_DL":"DL.psi","dx_shape_DL":"DL.dx_psi","dX_shape_DL":"DL.dX_psi","dS_shape_DL":"DL.dX_psi",
          "d2x_shape_DL":"DL.d2x_psi","d2S_shape_DL":"DL.d2x_psi"}
findings=collections.Counter(); examples={}; nscan=0; nskip=0; nofunc=collections.Counter()

def functions(lines):
    """Yield (name, first, last) of every top-level {...} block, by brace depth."""
    depth=0; start=None; name=None
    for i,l in enumerate(lines):
        if depth==0 and start is None and l.strip() and not l.lstrip().startswith("//"):
            m=re.search(r"(\w+)\s*\([^;]*$",l)
            if m: name=m.group(1)
        o=l.count("{"); c=l.count("}")
        if depth==0 and o: start=i
        depth+=o-c
        if start is not None and depth<=0:
            yield (name or "?",start,i); start=None; name=None; depth=0

for rel in files:
    try: lines=open(os.path.join(ROOT,rel),errors="replace").read().split("\n")
    except OSError: continue
    text="\n".join(lines)
    if "functable->" not in text:
        nskip+=1; continue
    nscan+=1
    setflags=collections.defaultdict(set)
    for m in SET_RE.finditer(text):
        d=CH.get(m.group(2))
        if d is None: continue
        setflags[m.group(1)].add((d,m.group(3),m.group(4)))
    depth=[0]*len(lines); c=0
    for i,l in enumerate(lines):
        depth[i]=c; c+=l.count("{")-l.count("}")
    for fname,f0,f1 in functions(lines):
        body=lines[f0:f1+1]
        structs=sorted(set(USE_RE.findall("\n".join(body))))
        if not structs:
            # Non-integrated evaluators do not fill the buffer themselves; the C++ side does, with the
            # struct named after them (elements_shapeinfo.cpp / elements.cpp).
            for pre,st in (("GetZ2Fluxes","Z2Fluxes"),("EvalLocalExpression","LocalExprs"),
                           ("EvalExtremumExpression","ExtremumExprs"),("EvalIntegralExpression","IntegralExprs"),
                           ("TracerAdvection","TracerAdvection")):
                if fname.startswith(pre): structs=[st]; break
        if len(structs)!=1:
            if any(READ_RE.search(l) and any(b in l for b in BUF2FLAG) for l in body):
                nofunc[(fname.split(str(0))[0], len(structs))]+=1
            continue
        flags=setflags[structs[0]]
        cat=re.sub(r"\[\d+\]","",structs[0])
        def report(d,buf,sp):
            f=BUF2FLAG[buf]
            ok=(d,None,f) in flags if ("." in f) else (d,sp,f) in flags
            if not ok:
                k=(cat,buf,f,sp or ""); findings[k]+=1; examples.setdefault(k,rel+":"+fname)
        for i,l in enumerate(body):
            gi=f0+i
            isdecl=("SHAPE_FUNCTION_DECL(" in l) or bool(re.search(r"double const \* \w+ =",l))
            if isdecl:
                m=re.search(r"(?:SHAPE_FUNCTION_DECL\((\w+)\)|double const \* (\w+))\s*=\s*(.+?);",l)
                if not m: continue
                nm=m.group(1) or m.group(2)
                mm=READ_RE.search(m.group(3))
                if not mm or mm.group(2) not in BUF2FLAG: continue
                d=mm.group(1).count("shapeinfo")-1; buf=mm.group(2)
                sp=next((x for x in IDX.findall(mm.group(3)) if x.startswith("SPACE_INDEX_")),None)
                j=gi+1; d0=depth[gi]; pat=re.compile(r"\b%s\s*\["%re.escape(nm))
                while j<=f1 and depth[j]>=d0:
                    if pat.search(lines[j]): report(d,buf,sp); break
                    j+=1
                continue
            for mm in READ_RE.finditer(l):
                buf=mm.group(2)
                if buf not in BUF2FLAG: continue
                d=mm.group(1).count("shapeinfo")-1
                sp=next((x for x in IDX.findall(mm.group(3)) if x.startswith("SPACE_INDEX_")),None)
                report(d,buf,sp)
print("files scanned: %d (skipped %d partial generations)"%(nscan,nskip))
if not findings: print("  NO under-request found (every dereferenced family is flagged by the struct the function passes)")
for k,v in sorted(findings.items(),key=lambda kv:-kv[1]):
    print("  %-14s reads %-14s needs %-12s %-16s : %5d  e.g. %s"%(k[0],k[1],k[2],k[3],v,examples[k]))
if nofunc:
    print("  (functions with 0 or >1 required-shapes struct, not attributable:)")
    for k,v in sorted(nofunc.items(),key=lambda kv:-kv[1])[:10]: print("     %s: %d"%(k,v))
