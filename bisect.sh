#!/usr/bin/env bash
cd ~/code/pyoomph 
./clean all; 
./prebuild.sh || exit 125  
./build_for_develop.sh || exit 125

echo "crashed" > indicator.txt
python3 buggy_code.py || echo 
STATUS=$(cat indicator.txt)
if [[ "$STATUS" == "okay" ]]; then
 exit 0
fi

if [[ "$STATUS" == "failed" ]]; then
 exit 1
fi

exit 125
