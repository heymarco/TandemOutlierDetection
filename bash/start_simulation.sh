#!/bin/bash
while getopts t:r:R:w:e: flag
do
  case "${flag}" in
  t) type=${OPTARG};;
  r) rounds=${OPTARG};;
  w) write=${OPTARG};;
  e) exp=${OPTARG};;
  R) reps=${OPTARG};;
esac
done

num_clients=0
if [ "$type" == "powertool" ];
then
  num_clients=15
elif [ "$type" == "local/global" ];
then
  num_clients=30
elif [ "$type" == "partition_outlier" ];
then
  num_clients=30
fi

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

python simulation/start_server.py -num_rounds "$rounds" -to_csv "$write" -type "$type" -exp_index "$exp" -num_reps "$reps" -num_clients "$num_clients" &
sleep 2

for i in $(seq 0 $((num_clients-1))); do
  python simulation/start_client.py -type "$type" -client_index "$i" -exp_index "$exp" -num_reps "$reps" &
  pids[$((i+1))]=$!
  done

# wait for all pids
for pid in ${pids[*]}; do
  wait $pid
done
