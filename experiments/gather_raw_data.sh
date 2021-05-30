#! /bin/bash

set -e

# Operate out of the script directory
SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
cd $SCRIPT_PATH

if [ -z "$1" ]; then
  echo "Usage: $0 <device>"
  exit 1
fi

python3 run_experiment.py $1
tar cvzf hv2-$1.tar.gz *.csv
rm *.csv
