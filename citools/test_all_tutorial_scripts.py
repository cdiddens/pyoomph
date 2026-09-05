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

from pathlib import Path
import sys,os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--quick-test", help="Stops after the first successful Newton method. Useful for quick testing", action="store_true")
parser.add_argument("--tcc", help="Used TCC", action="store_true")
parser.add_argument("--no-petsc", help="Do not select a PETSc build at all: skip the $PETSC_DIR/$PETSC_ARCH_REAL/$PETSC_ARCH_COMPLEX check and run every script with the inherited PYTHONPATH", action="store_true")
parser.add_argument("--keep-logs", help="Keep log files also of successful tests", action="store_true")
parser.add_argument("--keep-outdirs", help="Do not delete each script's output directory after it runs. Useful for comparing generated code across repeated runs (e.g. for determinism testing)", action="store_true")
parser.add_argument("--timeout", type=float, default=0.0, metavar="SECONDS", help="Kill a script that has run this long and report it as a failure. Default 0, i.e. wait forever. Meant for unattended runs: one script that hangs otherwise takes the whole pass with it (a macOS-arm64 wheel spent 2 hours in Accelerate's sparse solver before segfaulting)")
parser.add_argument("--report-json", metavar="PATH", default=None, help="Also write one JSON record per script (status, wall time, simulation time) to PATH. Meant for CI, which turns it into a table; the printed output is unchanged")
parser.add_argument("--mpirun", type=int, default=0, metavar="N", help="Run each script under 'mpirun -n N' instead of directly. Default 0, i.e. no mpirun")
parser.add_argument("--omp", type=int, default=0, metavar="N", help="Pass '--omp N' to each script, i.e. assemble the elements on N threads. Default 0, i.e. leave each script on the serial element loop. Composes with --mpirun, which is threads per rank then")
parser.add_argument("--distribute", help="Pass --distribute to each script, i.e. distribute the mesh over the ranks. Only meaningful together with --mpirun", action="store_true")
# Deliberately general rather than one option per solver. Note what forcing a solver DOES, though:
# it changes what is being computed, not just how fast. petsc_mumps collapses hopf_switch's arclength
# continuation and plain --petsc (an iterative KSP) fails outright on the augmented systems, so a
# failure under this flag is not by itself a failure of the backend - re-run the same script without
# it before concluding anything.
parser.add_argument("--extra-arg", action="append", default=[], metavar="ARG", help="Pass this argument on to every script. Repeatable, e.g. --extra-arg --mumps. See the note in the source about what forcing a solver changes")
# For chasing one flaky script across the platforms: a full pass is hours per OS, and a script that
# only fails once in a while has to be run as the CI runs it - same wheel, same runner - not once in
# a scratch directory. Substring rather than glob, on "Folder/script.py", so that a bare script name
# is the common case and a folder name still selects the folder.
parser.add_argument("--only", action="append", default=[], metavar="SUBSTRING", help="Run only the scripts whose 'Folder/script.py' path contains one of these substrings. Repeatable. Folders with no match are skipped entirely")
# Folders to skip used to be read from sys.argv directly, which stopped working when argparse was
# added - argparse rejects any positional argument it does not know about.
parser.add_argument("skips", nargs="*", help="Bundle folders to skip, e.g. Temporal_ODEs")
args = parser.parse_args()

# Resolved before the chdir below, so that a relative --report-json still lands where the caller
# meant it and not somewhere inside the exported bundle.
report_json=Path(args.report_json).resolve() if args.report_json else None

os.chdir(Path(__file__).parent)

import glob,re,subprocess,time
import shutil

# Every run stamps "Elapsed time: <h:mm:ss.ffffff>" into its log file when the Problem is released
# (Problem._write_log_footer). That is the number worth comparing between a serial and an mpirun
# pass, or between two linear solvers: it starts at initialise() and so covers the problem setup,
# the code generation and compilation and every solve, but not the interpreter start-up and the
# imports, which are a large and constant part of the subprocess's wall time. It is harvested right
# after each script, because the output directory holding it is deleted again a few lines later.
_LOGFILE_NAME="_pyoomph_logfile.txt" # Problem.logfile_name
_ELAPSED_LINE=re.compile(r"^Elapsed time:\s*(.+?)\s*$")

def elapsed_seconds(logfile):
  """The seconds behind the "Elapsed time:" footer of one pyoomph log file, or None."""
  try:
    with open(logfile,"rb") as lf:
      lf.seek(0,os.SEEK_END)
      lf.seek(max(0,lf.tell()-4096)) # the footer is the last handful of lines
      tail=lf.read().decode("utf-8",errors="replace")
  except OSError:
    return None
  for line in reversed(tail.splitlines()):
    m=_ELAPSED_LINE.match(line.strip())
    if m is None:
      continue
    text=m.group(1)
    days=0.0
    if "day" in text: # str(timedelta) spells the days out: "1 day, 0:02:03.4"
      head,_,text=text.partition(",")
      try:
        days=float(head.split()[0])
      except (IndexError,ValueError):
        return None
      text=text.strip()
    parts=text.split(":")
    if len(parts)!=3:
      return None
    try:
      return days*86400.0+int(parts[0])*3600.0+int(parts[1])*60.0+float(parts[2])
    except ValueError:
      return None
  return None

def simulation_seconds(started_at):
  """(seconds, number of log files) over every pyoomph run the script just finished did.

  Summed rather than taken from one file: a script may build several problems one after another,
  and parallel_running.py even starts further scripts of its own under mpirun.

  Only files written by this run count. Output directories of earlier scripts in the same folder
  can still be lying around - only the one named after the script is deleted afterwards, and
  --keep-outdirs keeps even that - and their footers would otherwise be added to every later
  script's time.
  """
  total,found=0.0,0
  for logfile in Path(".").rglob(_LOGFILE_NAME):
    try:
      if logfile.stat().st_mtime<started_at-1.0:
        continue
    except OSError:
      continue
    secs=elapsed_seconds(logfile)
    if secs is not None:
      total+=secs
      found+=1
  return (total if found else None),found

# Third-party packages a few tutorial bundles need that a plain pyoomph install does not pull in.
# A script that dies on the import of one of these has not regressed - the machine simply cannot run
# it - and reporting that as a failure every night is how a failure list stops being read. So it is
# reported as a skip instead, by name, and listed again at the end where the nightly picks it up.
# Nothing else is forgiven: a ModuleNotFoundError for anything not on this list is a real failure.
_OPTIONAL_MODULES={"precice":"preCICE (pip install pyprecice, plus a libprecice built for this "
                             "distribution's release)",
                   "shapely":"shapely (pip install pyoomph[topology]), which plans the axisymmetric "
                             "pinch-off and coalescence surgery",
                   }
# petsc4py/slepc4py used to be forgiven here as well, because the eigen scripts spelled out
# set_eigensolver("slepc") and could not run without them. None of them does any more: they leave the
# choice to pyoomph's autodetection, which takes SLEPc where PETSc is installed (and it is still the
# only backend that solves an eigenproblem distributed) and the built-in "spectra" otherwise. So a
# missing PETSc is no longer a reason to skip anything - on a --no-petsc run, i.e. the GitHub workflow
# testing a plain wheel, these scripts have to RUN and produce the same answers, which is the point of
# having a PETSc-free eigensolver in the first place. A ModuleNotFoundError for petsc4py is therefore a
# real failure again.
# Scripts that open a window and wait for the user. tkinter's mainloop never returns on its own, so
# under this harness they would either hang forever or - on a headless nightly - die on "no display
# name and no $DISPLAY environment variable", neither of which says anything about pyoomph. They are
# skipped here and listed again at the end, alongside the preCICE reminder, because they still have
# to be run by hand before a release. The GUI's own logic is covered headlessly by the worker scripts
# in tests/ (tests/*_worker.py drive the model without ever mapping a window).
_MANUAL_GUI_SCRIPTS={"thin_film_bifurcation_gui.py":"opens the interactive bifurcation GUI"}

_MISSING_MODULE=re.compile(rb"ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'")
# pyoomph catches the ImportError of a solver backend and re-raises it as its own RuntimeError
# (solvers/generic.py _unavailable_solver_message), so the missing package is named in the middle of
# the last line rather than at the start of it. No tutorial script names a solver backend explicitly
# any more, so this no longer fires for the eigen scripts; it stays for any script that does select a
# backend whose package this machine happens not to have.
_UNAVAILABLE_SOLVER=re.compile(rb"^RuntimeError: .* is not available \(ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'\)")
_BROKEN_IMPORT=re.compile(rb"^ImportError: (.*)$")
_TRACEBACK_FRAME=re.compile(rb'^  File "([^"]+)"')
# Under mpirun every stderr line is tagged "[rank N] " - on every rank and in every --mpi-output mode,
# see pyoomph/generic/logging.py _ConsoleWrapper._decorate. The three anchored patterns above would
# therefore never match in the MPI pass (only _MISSING_MODULE, which searches, survived), so the tag
# comes off before a line is examined.
_RANK_TAG=re.compile(rb"^\[rank \d+\] ?")

def untag_rank(line):
  return _RANK_TAG.sub(b"",line)

def missing_optional_module(output):
  """(package, why) when an optional package, and nothing else, ended this script - else None.

  Only the LAST line is examined, i.e. the exception the script actually died of. A
  ModuleNotFoundError caught and handled somewhere in the middle of a run says nothing about why it
  failed later on, and matching that would turn an unrelated crash into a silent skip.

  Two ways an optional package can be unusable, and both have been seen on duarte:

    not installed  - ModuleNotFoundError, the obvious one.
    installed but unloadable - the bindings import and then die on a shared library. duarte ran for
        weeks with libprecice3 3.4.1 unpacked from the *resolute* .deb on *noble*: dpkg accepts it,
        because the Depends are unversioned enough, but libprecice.so.3 then wants boost 1.90,
        glibc 2.43 and libpython3.14 and cannot be loaded at all. pyprecice imports and dies with
        "libboost_log_setup.so.1.90.0: cannot open shared object file". Fixed there by installing
        the noble build (see dev_docs/precice_setup.md); the branch stays for the next machine.

  The second is still "this machine cannot run preCICE", but a bare ImportError test would also
  swallow a broken pyoomph extension, so it counts only when a traceback frame is inside the
  optional package itself - i.e. the import that failed was that package's own.
  """
  lines=output.strip().splitlines()
  if not lines:
    return None
  last=untag_rank(lines[-1])
  m=_UNAVAILABLE_SOLVER.match(last)
  if m is not None:
    name=m.group(1).decode().split(".")[0]
    return (name,"is not installed here") if name in _OPTIONAL_MODULES else None
  m=_MISSING_MODULE.search(last)
  if m is not None:
    name=m.group(1).decode().split(".")[0]
    return (name,"is not installed here") if name in _OPTIONAL_MODULES else None
  m=_BROKEN_IMPORT.match(last)
  if m is None:
    return None
  frames=[f.group(1).decode() for f in (_TRACEBACK_FRAME.match(untag_rank(l)) for l in lines) if f]
  for name in _OPTIONAL_MODULES:
    if any(("/"+name+"/") in p or p.endswith("/"+name+".py") for p in frames):
      return (name,"is installed but cannot be loaded here ("+m.group(1).decode()+")")
  return None

# Under --mpirun the ranks' output is followed by mpirun's own epilogue ("Primary job terminated
# normally...", "mpirun detected that one or more processes exited with non-zero status"), each in a
# block fenced by a line of dashes. missing_optional_module() looks at the last line only - see there
# for why - so without dropping that epilogue every optional-package skip would be counted as a real
# failure in the MPI pass, which is exactly the noise the skip mechanism exists to avoid.
_DASH_RULE=re.compile(rb"^-{20,}$")
# Not everything mpirun appends is fenced, and since 2026-08-18 pyoomph appends a line of its own.
# Both sit AFTER the traceback and so become "the last line" that missing_optional_module() reads:
#
#   pyoomph: uncaught ... -- aborting the whole job.
#       generic/mpi.py's excepthook (commit a8973de2), which turns a rank dying alone - and the other
#       three then spinning in a collective until the step's timeout - into an MPI_Abort. It prints
#       this note after the traceback it has just let through.
#   [duarte:37160] N more processes have sent help message ...
#       Open MPI's help-message aggregator, written by mpirun itself once the job is down. Plain
#       "[host:pid] " lines, with no dash rules around them.
#
# Together they are why the mpirun pass reported rayleigh_plateau_pinchoff.py as FAILED on the
# 2026-09-02 and 2026-09-03 nightlies while the serial pass skipped it for the same missing shapely.
_UNFENCED_EPILOGUE=(re.compile(rb"^\[[^\]\s]+:\d+\]\s"),
                    re.compile(rb"^pyoomph: uncaught \w+ on MPI rank \d+ of \d+ -- aborting the whole job\."))

def strip_mpirun_epilogue(output):
  lines=output.rstrip().splitlines()
  peeled=True
  while peeled: # the two kinds interleave, so keep going until a pass changes nothing
    peeled=False
    while len(lines)>=2 and _DASH_RULE.match(lines[-1].strip()):
      for i in range(len(lines)-2,-1,-1):
        if _DASH_RULE.match(lines[i].strip()):
          del lines[i:]
          peeled=True
          break
      else:
        break # an unpaired rule: not one of mpirun's blocks, leave the output alone
      while lines and not lines[-1].strip():
        lines.pop()
    while lines and any(p.match(untag_rank(lines[-1].strip())) for p in _UNFENCED_EPILOGUE):
      lines.pop()
      peeled=True
    while lines and not lines[-1].strip():
      lines.pop()
  return b"\n".join(lines)

# The bundle of tutorial scripts is no longer a committed zip file - it is assembled from the
# tutorial sources, both here and during the documentation build (see docs/source/conf.py).
# So this pipeline tests the scripts as they currently are in the tree.
sys.path.insert(0, str(Path("../docs/source/tutorial").resolve()))
import tutorial_bundle


# In a subprocess, deliberately: importing petsc4py here would call MPI_Init in *this* process, and
# PETSc's MPI_Init sets some twenty PMIX_*/OMPI_* variables via C setenv(). Python's os.environ never
# sees those, but exec() hands the real C environ to the tested scripts, where the mpirun of
# --mpirun finds PMIX_NAMESPACE/PMIX_RANK, concludes it is already running inside an MPI job and
# exits 1 without printing anything at all - i.e. every single script "failing" with an empty log.
# Keeping this process free of MPI is what makes --mpirun work; nothing here needs PETSc itself.
# The reason is printed, not just the verdict: "petsc4py is not importable" is true of a directory
# that is not there, of one built for another Python, and of one whose libpetsc cannot be found - and
# those are three different things to go and fix. The macOS arm64 tutorial job of 30th August 2026
# reported the bare verdict for a petsc4py whose DIRECTORY the harness had already located, which
# left the actual ImportError - the only part that says which of the three it is - unsaid.
_PETSC_PROBE="""
import sys
try:
  from petsc4py import PETSc
except ImportError as e:
  print("NO_PETSC4PY")
  print("REASON:", e, file=sys.stderr)
  raise SystemExit
# slepc4py, and not only petsc4py, because PETSc.ScalarType below is a compile-time constant of the
# petsc4py extension and says nothing about the libpetsc that was actually mapped. A complex
# petsc4py that loaded the real libpetsc - which is what a DYLD_LIBRARY_PATH pointing at the other
# arch produces - answers COMPLEX here and is nonetheless wrong. Loading SLEPc is what catches it:
# the two builds do not export the same symbols, so the arch mismatch becomes an ImportError instead
# of a silently wrong sizeof(PetscScalar). It is also what pyoomph's own autodetection imports, so a
# tree that fails here is a tree that would have fallen back to a slower solver without saying so.
try:
  from slepc4py import SLEPc
except ImportError as e:
  print("NO_SLEPC4PY")
  print("REASON:", e, file=sys.stderr)
  raise SystemExit
import numpy
print("COMPLEX" if PETSc.ScalarType is numpy.complex128 else "NOT_COMPLEX")
"""

# Which PETSc a script gets is decided per script, not once for the whole run. The complex build is
# the slower of the two - every scalar is two doubles - and it is not what a user running an ordinary
# tutorial has loaded, so testing everything against it tests a configuration nobody uses. Only the
# scripts that genuinely need it get it: the azimuthal / Cartesian mode decomposition assembles a
# complex Jacobian, and a real PETSc cannot hold it.
#
# Recognised by the call that switches the mode decomposition on. The negative lookahead keeps
# setup_for_stability_analysis(azimuthal_stability=False) - which utils/periodic_driving_response.py
# passes explicitly - on the real build where it belongs.
_COMPLEX_NEEDED=re.compile(r"(?:azimuthal_stability|additional_cartesian_mode)\s*=\s*(?!False\b)")

# The periodic-orbit scripts have no such marker in their source - they need the complex build for
# what they solve rather than for how the Jacobian is assembled - so they are named outright: the
# Hopf tracker and the Floquet multipliers of an orbit are complex quantities.
_COMPLEX_BY_NAME={"langford_floquet.py","hopf_switch.py","langford_time_integration.py"}

def needs_complex_petsc(script):
  if Path(script).name in _COMPLEX_BY_NAME:
    return True
  return _COMPLEX_NEEDED.search(Path(script).read_text(errors="replace")) is not None

def petsc_pythonpath(arch,varname):
  """The directory holding petsc4py for one of the two builds, from $PETSC_DIR/$arch."""
  root=Path(os.environ["PETSC_DIR"])/arch
  # PETSc's own build installs it as $PETSC_DIR/$PETSC_ARCH/lib/petsc4py, but pointing the variable
  # straight at that lib directory is just as sensible a thing for a user to have done.
  candidates=[root/"lib",root]
  for cand in candidates:
    if (cand/"petsc4py").is_dir():
      return cand
  raise FileNotFoundError("No petsc4py found for $%s=%s - looked in %s"%(varname,arch,", ".join(str(c) for c in candidates)))

def env_with_petsc(petscdir):
  env=dict(os.environ)
  # Prepended, not replaced: PYTHONPATH is where the tutorial bundle's own helper modules can live.
  env["PYTHONPATH"]=os.pathsep.join([str(petscdir)]+([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
  # The dynamic loader has to follow PYTHONPATH to the same arch, and this is not housekeeping: the
  # two builds ship the SAME leaf names (libpetsc.dylib, libslepc.dylib), and DYLD_LIBRARY_PATH is
  # searched by leaf name before an install_name or an RPATH. With only PYTHONPATH moved, the macOS
  # arm64 job's DYLD_LIBRARY_PATH stayed at env.sh's real/lib and the COMPLEX petsc4py mapped
  # real/lib/libpetsc - while still reporting ScalarType complex128, because that is a compile-time
  # constant of the extension module rather than a property of the library it loaded. slepc4py then
  # died on a symbol its real-scalar twin does not export ("Symbol not found: _NEPCISSGetExtraction"),
  # pyoomph's _have_petsc_mumps() swallowed the ImportError and fell back to accelerate, and
  # rayleigh_benard_azimuthal_stability.py spent 55 s per LU factorisation until the 1800 s timeout.
  # The scripts that do NOT reach SLEPc were the worse half of that: they ran on a libpetsc with the
  # wrong sizeof(PetscScalar) and said nothing at all.
  # Prepended rather than replaced for the same reason as PYTHONPATH - the inherited value carries
  # the MPI runtime's lib directory, which both arches need.
  for var in ("DYLD_LIBRARY_PATH","LD_LIBRARY_PATH"):
    env[var]=os.pathsep.join([str(petscdir)]+([env[var]] if env.get(var) else []))
  return env

def check_petsc(env,arch,varname,want_complex):
  probe=subprocess.run([sys.executable,"-c",_PETSC_PROBE],capture_output=True,text=True,env=env)
  verdict=probe.stdout.split()
  where="$%s=%s (%s)"%(varname,arch,env["PYTHONPATH"].split(os.pathsep)[0])
  if "NO_PETSC4PY" in verdict:
    raise ImportError("petsc4py is not importable from %s by %s:\n%s"
                      %(where,sys.executable,(probe.stderr or "").strip() or "(the probe said nothing)"))
  if "NO_SLEPC4PY" in verdict:
    raise ImportError("petsc4py imports from %s but slepc4py does not, so pyoomph would silently fall "
                      "back to a slower solver.\nA 'Symbol not found' here means the loader took the "
                      "OTHER arch's libslepc: check that $DYLD_LIBRARY_PATH/$LD_LIBRARY_PATH point at "
                      "this arch and not at the one env.sh happens to set.\n%s"
                      %(where,(probe.stderr or "").strip() or "(the probe said nothing)"))
  wanted="COMPLEX" if want_complex else "NOT_COMPLEX"
  if wanted in verdict:
    return
  if ("NOT_COMPLEX" if want_complex else "COMPLEX") in verdict:
    raise AssertionError("The PETSc at %s has %s scalars, but %s is supposed to be the %s build"%(where,"real" if want_complex else "complex",varname,"complex" if want_complex else "real-scalar"))
  # Neither answer: petsc4py imported and then died (a mismatched libpetsc aborts here). Say that,
  # rather than blaming complex support for a crash the probe's own output already explains.
  raise RuntimeError("Could not determine the scalar type of the PETSc at %s - the check exited with %d and said:\n%s"%(where,probe.returncode,(probe.stderr or probe.stdout).strip()))

env_real=env_complex=None
if not args.no_petsc:
  missing=[v for v in ("PETSC_DIR","PETSC_ARCH_REAL","PETSC_ARCH_COMPLEX") if not os.environ.get(v)]
  if missing:
    raise EnvironmentError("Set %s to select the two PETSc builds (e.g. PETSC_DIR=~/code/petsc, "
                           "PETSC_ARCH_REAL=pyoomph_petsc_arch_real, "
                           "PETSC_ARCH_COMPLEX=pyoomph_petsc_arch_complex), or pass --no-petsc to run "
                           "with whatever PYTHONPATH already provides."%(", ".join("$"+v for v in missing)))
  env_real=env_with_petsc(petsc_pythonpath(os.environ["PETSC_ARCH_REAL"],"PETSC_ARCH_REAL"))
  env_complex=env_with_petsc(petsc_pythonpath(os.environ["PETSC_ARCH_COMPLEX"],"PETSC_ARCH_COMPLEX"))
  check_petsc(env_real,os.environ["PETSC_ARCH_REAL"],"PETSC_ARCH_REAL",want_complex=False)
  check_petsc(env_complex,os.environ["PETSC_ARCH_COMPLEX"],"PETSC_ARCH_COMPLEX",want_complex=True)
  print("Real-scalar PETSc:",env_real["PYTHONPATH"].split(os.pathsep)[0])
  print("Complex PETSc:    ",env_complex["PYTHONPATH"].split(os.pathsep)[0],"(normal-mode stability and periodic-orbit scripts only)")


problems=tutorial_bundle.check_consistency()
for problem in problems:
  print("PROBLEM:",problem)

if Path("pyoomph_tutorial_scripts").exists():
  print("Removing old pyoomph_tutorial_scripts folder")

bundle=tutorial_bundle.export_tree(Path("."))
print("Gathered",len(tutorial_bundle.collect()),"tutorial files into",bundle)

os.chdir("pyoomph_tutorial_scripts")
basedir=Path(".").absolute()

all_okay=not problems # inconsistent duplicated scripts already count as a failure

skips=args.skips
skipped_for_missing=[]
manual_gui=[]      # (folder, script) of the GUI scripts nobody can test unattended
timings=[]  # (folder, script, seconds, number of log files, whether the script failed)
untimed=[]  # ran, but left no "Elapsed time" footer anywhere
records=[]  # one dict per script, for --report-json


def folder_label(d):
  """The bundle folder's plain name, from what glob returned.

  glob("./*/") gives ".\\Advanced_Linear_Dynamics\\" on Windows, and the strip("/") that used to
  do this left the backslashes in - so the GitHub summary listed every script as
  "\\Advanced_Linear_Dynamics\\/rivulet.py" and no folder name a user typed matched the skip list.
  The separator is normalised rather than left to Path: this then gives the same answer when the
  report of a Windows run is read back on another machine.
  """
  return Path(d.replace("\\","/")).name

def wanted(folder,script):
  """Whether --only selects this script. No --only means everything, as before."""
  if not args.only:
    return True
  return any(o in folder+"/"+script for o in args.only)


for d in glob.glob("./*/"):
  if d in skips or folder_label(d) in skips:
    print("SKIPPING",d)
    continue
  if args.only and not any(wanted(folder_label(d),f.name)
                           for f in (basedir/d).glob("*.py")):
    # Silent: with --only naming one script, saying so for the other ten folders is noise.
    continue
  
  folder_okay=True
  os.chdir(basedir/d)
  print("TESTING FOLDER",d )
  for f in glob.glob("*.py"):
    if not wanted(folder_label(d),f):
      continue
    if f=="bifurcation_fold_param_change.py":
      # This is meant to crash when the parameter is changed, so it is not a regression test.
      continue
    if f in _MANUAL_GUI_SCRIPTS:
      print("   SKIPPING",f,"--",_MANUAL_GUI_SCRIPTS[f],"and must be checked manually")
      manual_gui.append((folder_label(d),f))
      records.append({"folder":folder_label(d),"script":f,"status":"skipped","note":_MANUAL_GUI_SCRIPTS[f]+", must be checked manually"})
      continue
    if args.mpirun>0 and f=="parallel_running.py":
      # This one spawns its own mpirun, so launching it under one already gives nested MPI.
      print("   SKIPPING",f,"-- it is just as spawner of other scripts")
      records.append({"folder":folder_label(d),"script":f,"status":"skipped","note":"only a spawner of other scripts, and this pass is already under mpirun"})
      continue
    env=None if args.no_petsc else (env_complex if needs_complex_petsc(f) else env_real)
    print("   Testing",f,"-- with the complex PETSc" if env is not None and env is env_complex else "")
    cmd=[sys.executable, '-u', f]
    if args.mpirun>0:
      cmd=["mpirun","-n",str(args.mpirun)]+cmd
    if args.quick_test:
      cmd.append("--quick-test")
    if args.tcc:
      cmd.append("--tcc")
    if args.distribute:
      cmd.append("--distribute")
    if args.omp>0:
      cmd+=["--omp",str(args.omp)]
    cmd+=args.extra_arg
    started_at=time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    #proc = subprocess.Popen([sys.executable, '-u', f], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    timed_out=False
    try:
      (stdout,_) = proc.communicate(timeout=args.timeout if args.timeout>0 else None)
    except subprocess.TimeoutExpired:
      # kill(), not terminate(): a script wedged inside a native solver does not necessarily get
      # around to running a Python signal handler. The output written so far is still wanted - it
      # is the only clue as to where it got stuck - so read it after the process is gone.
      timed_out=True
      proc.kill()
      (stdout,_) = proc.communicate()
    # The whole subprocess, i.e. interpreter start-up and imports included. Not comparable to the
    # "simulation time" below, but it is the only number a script that never built a Problem has.
    wall=time.time()-started_at
    secs,numlogs=simulation_seconds(started_at)
    optional=None
    if proc.returncode!=0 and not timed_out:
      # A killed script's last line is wherever it happened to be, so the optional-package test
      # would be reading tea leaves - and a timeout is never something to forgive anyway.
      optional=missing_optional_module(strip_mpirun_epilogue(stdout) if args.mpirun>0 else stdout)
    if optional is not None:
      mod,why=optional
      print("   SKIPPED",f,"--",_OPTIONAL_MODULES[mod],why)
      skipped_for_missing.append((folder_label(d),f,mod,why))
      status,note="skipped","needs "+_OPTIONAL_MODULES[mod]+", which "+why
    elif proc.returncode!=0:
      logf=Path(f).stem+".log"
      print(" ================= %s"%("TIMED OUT after %.0f s"%args.timeout if timed_out else "FAILED"),f,"see log at ",logf)
      with open(logf,"wb") as lf:
        lf.write(stdout)
      folder_okay=False
      status,note="failed",("timed out after %.0f s and was killed, see %s"%(args.timeout,logf)
                            if timed_out else "exited with %d, see %s"%(proc.returncode,logf))
    elif args.keep_logs:
      logf=Path(f).stem+".log"
      with open(logf,"wb") as lf:
        lf.write(stdout)
      status,note="passed",""
    else:
      status,note="passed",""

    records.append({"folder":folder_label(d),"script":f,"status":status,"note":note,
                    "wall_seconds":wall,"sim_seconds":secs,"num_logs":numlogs})

    if secs is not None:
      print("      simulation time: %.2f s"%secs+(" (%d runs)"%numlogs if numlogs>1 else ""))
      timings.append((folder_label(d),f,secs,numlogs,proc.returncode!=0))
    elif optional is None:
      # A script that never got as far as building a Problem, or one that was killed hard enough
      # to skip the atexit footer. Named at the end so the total is not silently short.
      untimed.append((folder_label(d),f))

    if not args.keep_outdirs:
      shutil.rmtree(Path(f).stem,ignore_errors=True)

  if folder_okay:
    print("ALL OKAY in",d)
    print()
  else:
    print("SOME TESTS FAILED in",d)
    print()
    all_okay=False

# Before the verdict, so that a green run still says out loud what it did not cover. The nightly
# parses these lines (see citools/nightly_develop.sh) and puts them in its Coverage section.
if skipped_for_missing:
  print()
  print("SKIPPED FOR MISSING OPTIONAL DEPENDENCIES:")
  for folder,script,mod,why in skipped_for_missing:
    print("   %s/%s needs %s, which %s"%(folder,script,_OPTIONAL_MODULES[mod],why))
  print()

if manual_gui:
  print()
  print("NOT RUN, CHECK MANUALLY (interactive GUI scripts):")
  for folder,script in manual_gui:
    print("   %s/%s %s"%(folder,script,_MANUAL_GUI_SCRIPTS[script]))
  print()

# Slowest first, since that is the end of the list one reads. The fixed "TIME" prefix is what the
# nightly greps for (see citools/nightly_develop.sh), so keep the column layout if you touch this.
if timings:
  print()
  print("SIMULATION TIMES (the \"Elapsed time\" each run recorded in its %s):"%_LOGFILE_NAME)
  for folder,script,secs,numlogs,failed in sorted(timings,key=lambda t:-t[2]):
    extra="".join([" (%d runs)"%numlogs if numlogs>1 else "", " (FAILED)" if failed else ""])
    print("   TIME %10.2f s  %s/%s%s"%(secs,folder,script,extra))
  print("   TIME TOTAL %.2f s over %d script(s)"%(sum(t[2] for t in timings),len(timings)))
  if untimed:
    print("   TIME MISSING for %d script(s): %s"%(len(untimed),", ".join(folder+"/"+script for folder,script in untimed)))
  print()

if all_okay:
  print("ALL TESTS PASSED -- But please check e.g. preCICE and the interactive GUI scripts manually")
else:
  print("SOME TESTS FAILED")

# Written last, and only on request: the printed report above stays the interface for a human and
# for the nightly's greps, this one is for a machine (citools/tutorial_report_to_summary.py turns it
# into the GitHub job summary). The exit code stays 0 either way - callers that want a gate look at
# "all_okay" here, or at the report.
if report_json is not None:
  import json,platform
  report_json.parent.mkdir(parents=True,exist_ok=True)
  with open(report_json,"w") as jf:
    json.dump({"all_okay":all_okay,
               "bundle_problems":problems,
               "platform":platform.platform(),
               "python":sys.version.split()[0],
               "options":{"quick_test":args.quick_test,"tcc":args.tcc,"no_petsc":args.no_petsc,
                          "mpirun":args.mpirun,"omp":args.omp,"distribute":args.distribute,
                          "extra_arg":args.extra_arg,
                          "timeout":args.timeout,
                          "skips":skips,"only":args.only},
               "scripts":records},jf,indent=1)
  print("Wrote the machine-readable report to",report_json)
  
  
