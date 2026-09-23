#!/usr/bin/env bash
# Re-run the Arcade experiment measurements. Each run trains from scratch; greedy eval every 5k steps.
set -e
cd "$(dirname "$0")"
node train_brain.js 50000 base '{}'                              > results/abl_base.txt
node train_brain.js 50000 notarget '{"useTarget":false}'         > results/abl_notarget.txt
node train_brain.js 50000 gamma05 '{"gamma":0.5}'                > results/abl_gamma05.txt
node train_brain.js 40000 noreplay '{"useReplay":false}'         > results/abl_noreplay.txt
node train_brain.js 40000 lr01 '{"lr":0.01}'                     > results/abl_lr01.txt
node train_brain.js 40000 eps0 '{"epsStart":0,"epsEnd":0}'       > results/abl_eps0.txt
echo "done: see results/abl_*.txt"
