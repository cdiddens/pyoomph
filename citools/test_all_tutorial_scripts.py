from pathlib import Path
import sys,os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--quick-test", help="Stops after the first successful Newton method. Useful for quick testing", action="store_true")
parser.add_argument("--tcc", help="Used TCC", action="store_true")
parser.add_argument("--no-petsc", help="Ignore PETSc check", action="store_true")
parser.add_argument("--keep-logs", help="Keep log files also of successful tests", action="store_true")
parser.add_argument("--keep-outdirs", help="Do not delete each script's output directory after it runs. Useful for comparing generated code across repeated runs (e.g. for determinism testing)", action="store_true")
# Folders to skip used to be read from sys.argv directly, which stopped working when argparse was
# added - argparse rejects any positional argument it does not know about.
parser.add_argument("skips", nargs="*", help="Bundle folders to skip, e.g. Temporal_ODEs")
args = parser.parse_args()

os.chdir(Path(__file__).parent)

import glob,re,subprocess
import shutil

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

# The bundle of tutorial scripts is no longer a committed zip file - it is assembled from the
# tutorial sources, both here and during the documentation build (see docs/source/conf.py).
# So this pipeline tests the scripts as they currently are in the tree.
sys.path.insert(0, str(Path("../docs/source/tutorial").resolve()))
import tutorial_bundle


if not args.no_petsc:
  try:
    from  petsc4py import PETSc
  except ImportError:
    raise ImportError("petsc4py not found, cannot run tests with eigenvalue solvers. Please install petsc4py and make sure it is in the PYTHONPATH")

  import numpy
  assert PETSc.ScalarType is numpy.complex128, "PETSc does not support complex numbers, cannot run tests with eigenvalue solvers. Please install a version of PETSc with complex support and make sure petsc4py is using that version."
  

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


for d in glob.glob("./*/"):
  if d in skips or d.strip("/").strip("./") in skips:
    print("SKIPPING",d)
    continue
  
  folder_okay=True
  os.chdir(basedir/d)
  print("TESTING FOLDER",d )
  for f in glob.glob("*.py"):
    if f=="bifurcation_fold_param_change.py":
      continue
    print("   Testing",f)  
    cmd=[sys.executable, '-u', f]
    if args.quick_test:
      cmd.append("--quick-test")
    if args.tcc:
      cmd.append("--tcc")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #proc = subprocess.Popen([sys.executable, '-u', f], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (stdout,_) = proc.communicate()
    optional=missing_optional_module(stdout) if proc.returncode!=0 else None
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

if all_okay:
  print("ALL TESTS PASSED -- But please check e.g. preCICE runs manually")
else:
  print("SOME TESTS FAILED")
  
  
