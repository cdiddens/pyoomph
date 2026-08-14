from pathlib import Path
import sys,os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--quick-test", help="Stops after the first successful Newton method. Useful for quick testing", action="store_true")
parser.add_argument("--tcc", help="Used TCC", action="store_true")
parser.add_argument("--no-petsc", help="Do not select a PETSc build at all: skip the $PETSC_DIR/$PETSC_ARCH_REAL/$PETSC_ARCH_COMPLEX check and run every script with the inherited PYTHONPATH", action="store_true")
parser.add_argument("--keep-logs", help="Keep log files also of successful tests", action="store_true")
parser.add_argument("--keep-outdirs", help="Do not delete each script's output directory after it runs. Useful for comparing generated code across repeated runs (e.g. for determinism testing)", action="store_true")
parser.add_argument("--mpirun", type=int, default=0, metavar="N", help="Run each script under 'mpirun -n N' instead of directly. Default 0, i.e. no mpirun")
parser.add_argument("--distribute", help="Pass --distribute to each script, i.e. distribute the mesh over the ranks. Only meaningful together with --mpirun", action="store_true")
# Folders to skip used to be read from sys.argv directly, which stopped working when argparse was
# added - argparse rejects any positional argument it does not know about.
parser.add_argument("skips", nargs="*", help="Bundle folders to skip, e.g. Temporal_ODEs")
args = parser.parse_args()

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
                             "distribution's release)"}
_MISSING_MODULE=re.compile(rb"ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'")
_BROKEN_IMPORT=re.compile(rb"^ImportError: (.*)$")
_TRACEBACK_FRAME=re.compile(rb'^  File "([^"]+)"')

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
  last=lines[-1]
  m=_MISSING_MODULE.search(last)
  if m is not None:
    name=m.group(1).decode().split(".")[0]
    return (name,"is not installed here") if name in _OPTIONAL_MODULES else None
  m=_BROKEN_IMPORT.match(last)
  if m is None:
    return None
  frames=[f.group(1).decode() for f in (_TRACEBACK_FRAME.match(l) for l in lines) if f]
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

def strip_mpirun_epilogue(output):
  lines=output.rstrip().splitlines()
  while len(lines)>=2 and _DASH_RULE.match(lines[-1].strip()):
    for i in range(len(lines)-2,-1,-1):
      if _DASH_RULE.match(lines[i].strip()):
        del lines[i:]
        break
    else:
      break # an unpaired rule: not one of mpirun's blocks, leave the output alone
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
_PETSC_PROBE="""
try:
  from petsc4py import PETSc
except ImportError:
  print("NO_PETSC4PY")
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
  return env

def check_petsc(env,arch,varname,want_complex):
  probe=subprocess.run([sys.executable,"-c",_PETSC_PROBE],capture_output=True,text=True,env=env)
  verdict=probe.stdout.split()
  where="$%s=%s (%s)"%(varname,arch,env["PYTHONPATH"].split(os.pathsep)[0])
  if "NO_PETSC4PY" in verdict:
    raise ImportError("petsc4py is not importable from "+where)
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
timings=[]  # (folder, script, seconds, number of log files, whether the script failed)
untimed=[]  # ran, but left no "Elapsed time" footer anywhere


for d in glob.glob("./*/"):
  if d in skips or d.strip("/").strip("./") in skips:
    print("SKIPPING",d)
    continue
  
  folder_okay=True
  os.chdir(basedir/d)
  print("TESTING FOLDER",d )
  for f in glob.glob("*.py"):
    if f=="bifurcation_fold_param_change.py":
      # This is meant to crash when the parameter is changed, so it is not a regression test.
      continue
    if args.mpirun>0 and f=="parallel_running.py":
      # This one spawns its own mpirun, so launching it under one already gives nested MPI.
      print("   SKIPPING",f,"-- it is just as spawner of other scripts")
      continue
    if args.mpirun>0 and (f=="deflated_solve.py" or f=="deflated_continuation.py"):
      # The deflation these two drive is a custom assembly handler, which has no MPI path yet.
      print("   SKIPPING",f,"-- custom assemblers not MPI capable yet")
      continue
    if args.mpirun>0 and not args.distribute and f=="cr_static_condensation.py":
      # Not a defect of the script: its selection pairs the bubble velocity (nodal) with the pressure
      # gradients (element-internal), and oomph-lib numbers every nodal value before any internal one,
      # so in a REPLICATED run the two halves of a block land on different ranks' rows and no rank can
      # eliminate it. pyoomph says exactly that and refuses (src/problem.cpp), and the tutorial says it
      # too. With --distribute the dofs are renumbered per rank and the script runs, so it is only this
      # pass that has to leave it out.
      print("   SKIPPING",f,"-- CR condensation needs --distribute under MPI, see the tutorial")
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
    started_at=time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    #proc = subprocess.Popen([sys.executable, '-u', f], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (stdout,_) = proc.communicate()
    secs,numlogs=simulation_seconds(started_at)
    optional=None
    if proc.returncode!=0:
      optional=missing_optional_module(strip_mpirun_epilogue(stdout) if args.mpirun>0 else stdout)
    if optional is not None:
      mod,why=optional
      print("   SKIPPED",f,"--",_OPTIONAL_MODULES[mod],why)
      skipped_for_missing.append((d.strip("/").strip("./"),f,mod,why))
    elif proc.returncode!=0:
      logf=Path(f).stem+".log"
      print(" ================= FAILED",f,"see log at ",logf)
      with open(logf,"wb") as lf:
        lf.write(stdout)
      folder_okay=False
    elif args.keep_logs:
      logf=Path(f).stem+".log"
      with open(logf,"wb") as lf:
        lf.write(stdout)

    if secs is not None:
      print("      simulation time: %.2f s"%secs+(" (%d runs)"%numlogs if numlogs>1 else ""))
      timings.append((d.strip("/").strip("./"),f,secs,numlogs,proc.returncode!=0))
    elif optional is None:
      # A script that never got as far as building a Problem, or one that was killed hard enough
      # to skip the atexit footer. Named at the end so the total is not silently short.
      untimed.append((d.strip("/").strip("./"),f))

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
  print("ALL TESTS PASSED -- But please check e.g. preCICE runs manually")
else:
  print("SOME TESTS FAILED")
  
  
