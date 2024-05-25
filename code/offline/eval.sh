#!/bin/bash


policies=(
  "TD3_BC"
  "BC"
  "memTD3"
  "memTD3 --no_bc"
)

envs=(
	"halfcheetah-random-v2"
	"hopper-random-v2"
	"walker2d-random-v2"
	"halfcheetah-medium-v2"
	"hopper-medium-v2"
	"walker2d-medium-v2"
	"halfcheetah-expert-v2"
	"hopper-expert-v2"
	"walker2d-expert-v2"
	"halfcheetah-medium-expert-v2"
	"hopper-medium-expert-v2"
	"walker2d-medium-expert-v2"
	"halfcheetah-medium-replay-v2"
	"hopper-medium-replay-v2"
	"walker2d-medium-replay-v2"
)
if command -v nvidia-smi &> /dev/null; then
    # Determine the correct query flag dynamically
    if nvidia-smi --help | grep -q -- "--query-gpu"; then
        query_flag="--query-gpu"
    elif nvidia-smi --help | grep -q -- "--query-gpus"; then
        query_flag="--query-gpus"
    else
        # Neither flag is found, default to "--query-gpus"
        query_flag="--query-gpus"
    fi
    # Get the GPU count
    gpu_count=$(nvidia-smi $query_flag=count --format=csv,noheader | wc -l)
else
    # nvidia-smi is not available, set GPU count to 0
    gpu_count=0
fi

i=0
to_execute=""
for ((seed=0;seed<5;seed+=1))
do
for env in ${envs[*]}
    do
    for policy in "${policies[@]}"
      do
      if [ $gpu_count -gt 0 ]; then
          device="--device cuda:$((i % gpu_count))"
      else
          device="--device cpu"
      fi
      i=$((i+1))
      to_execute+=" --env \"$env\" --policy $policy  $device --seed $seed;"
    done
  done
done
if [ -n "$gpu_count" ]; then
    p=$(($1 * gpu_count))
else
    p=$1
fi

echo "$to_execute" | tr ';' '\n' | xargs -P $p -I {} bash -c "python evaluate.py {}"

