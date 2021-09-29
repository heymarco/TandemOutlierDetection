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
if [ "$type" == "melbourne" ];
then
  num_clients=9
elif [ "$type" == "baiot" ];
then
  num_clients=9
elif [ "$type" == "arculus" ];
then
  num_clients=20
elif [ "$type" == "ipek" ];
then
  num_clients=15
elif [ "$type" == "mnist" ];
then
  num_clients=9
elif [ "$type" == "synth" ];
then
  num_clients=30
elif [ "$type" == "synth_vary_clients" ];
then
  repetitions_with_same_params=10
  all_num_clients=(2 3 4 6 8 12 15 20 25 30)
  counter_for_params=$((exp / repetitions_with_same_params))
  num_clients=${all_num_clients[$counter_for_params / 4]};
  echo "$((counter_for_params / 4))"
  echo "$num_clients"
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
