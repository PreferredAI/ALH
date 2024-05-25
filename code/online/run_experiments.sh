#!/bin/bash

# Script to reproduce results

gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader --id=0)
to_execute=""

envs=(
	"HalfCheetah-v3"
	"Hopper-v3"
	"Walker2d-v3"
	"Ant-v3"
	"Humanoid-v3"
	"Reacher-v2"
	"InvertedDoublePendulum-v2"
	"InvertedPendulum-v2"
	)
policies=(
  "memTD3" # ALH-g
  "memTD32" # ALH-a
  "memDDPG" # ALH-g (DDPG version)
  "memDDPG_adaptive" # ALH-a (DDPG version)
  "DDPG"
  "TD3"

)


i=0
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

for ((seed=0;seed<10;seed+=1))
  do
  for env in ${envs[*]}
    do
      for policy in ${policies[*]}
        do
          i=$((i+1))
          if [ $gpu_count -gt 0 ]; then
              device="--device cuda:$((i % gpu_count))"
          else
              device="--device cpu"
          fi
          to_execute+="--env \"$env\" --policy $policy --seed $seed $device;"
        done
    done
done

p=$(($1*$((gpu_count|| 1))))

echo "$to_execute" | xargs -d ";" -P $p -I {} bash -c "python main.py {}"
