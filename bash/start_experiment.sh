#!/bin/bash
while getopts t:r:R: flag
do
  case "${flag}" in
  t) type=${OPTARG};;
  r) rounds=${OPTARG};;
  R) reps=${OPTARG};;
esac
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

for i in $(seq 0 $((reps-1))); do
    bash bash/start_simulation.sh -t "$type" -r "$rounds" -e "$i" -w 1 -R "$reps"
    pids[$((i+1))]=$!
  done

# wait for all pids
for pid in ${pids[*]}; do
  wait $pid
done