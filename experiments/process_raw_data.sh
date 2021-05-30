#! /bin/bash

set -e

# Operate out of the script directory
SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
cd $SCRIPT_PATH

RESULTS_DIR="results/results-$(date "+%F_%H_%M")"

mkdir -p results
mkdir $RESULTS_DIR
mkdir $RESULTS_DIR/raw
mkdir $RESULTS_DIR/ops
mkdir $RESULTS_DIR/e2e
mkdir $RESULTS_DIR/archives

for archive in $(ls *.tar.gz); do
  tar xvzf $archive -C $RESULTS_DIR/raw
done

python3 process_results.py \
  --in-dir $RESULTS_DIR/raw \
  --out-ops $RESULTS_DIR/ops \
  --out-e2e $RESULTS_DIR/e2e

mv *.tar.gz $RESULTS_DIR/archives

