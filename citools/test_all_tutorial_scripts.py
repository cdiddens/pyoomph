from pathlib import Path
import sys,os

os.chdir(Path(__file__).parent)

import zipfile,glob,subprocess
import shutil



bundle= Path("../docs/source/tutorial/tutorial_example_scripts.zip")

with zipfile.ZipFile(str(bundle), 'r') as zipf:
    zipf.extractall(".")
    
os.chdir("pyoomph_tutorial_scripts")
basedir=Path(".").absolute()

all_okay=True

skips=sys.argv[1:]

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
    proc = subprocess.Popen([sys.executable, '-u', f], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (stdout,_) = proc.communicate()
    if proc.returncode!=0:
      logf=Path(f).stem+".log"
      print(" ================= FAILED",f,"see log at ",logf)
      with open(logf,"wb") as lf:
        lf.write(stdout)
      folder_okay=False
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
  
  
