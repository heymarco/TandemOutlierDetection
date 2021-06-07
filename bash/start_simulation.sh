#!/bin/bash
while getopts t:r:w:e: flag
do
  case "${flag}" in
  t) type=${OPTARG};;
  r) rounds=${OPTARG};;
  w) write=${OPTARG};;
  e) exp=${OPTARG};;
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
  num_clients=8
elif [ "$type" == "synth" ];
then
  num_clients=30
fi

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

python simulation/start_server.py -num_rounds "$rounds" -to_csv "$write" -type "$type" -exp_index "$exp" &
sleep 2

for i in $(seq 0 $((num_clients-1))); do
  python simulation/start_client.py -type "$type" -client_index "$i" &
  pids[$((i+1))]=$!
  done

# wait for all pids
for pid in ${pids[*]}; do
  wait $pid
done
