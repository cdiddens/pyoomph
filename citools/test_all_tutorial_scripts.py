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

import glob,subprocess
import shutil

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
    if proc.returncode!=0:
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

if all_okay:  
  print("ALL TESTS PASSED -- But please check e.g. preCICE runs manually")
else:
  print("SOME TESTS FAILED")
  
  
