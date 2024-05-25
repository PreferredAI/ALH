#!/bin/bash

# Script to reproduce results

to_execute=""

envs=(
	"MultiNormEnv"
	)
policies=(
  "--policy memTD32  --batch_size 48 --start_timesteps 1000  --hidden_dim 64 --hypo_dim 64 --max_timesteps 1000000 --eval_freq 5000  --mini_batch_size 24"
  "--policy memTD3  --batch_size 48 --start_timesteps 1000  --hidden_dim 64 --hypo_dim 64 --max_timesteps 1000000 --eval_freq 5000  --mini_batch_size 24"
  "--policy TD3  --batch_size 48 --start_timesteps 1000  --hidden_dim 64 --max_timesteps 1000000"
  "--policy TD3 -is_not_hard  --batch_size 48 --start_timesteps 1000  --hidden_dim 64 --max_timesteps 1000000"
)

i=0
seeds=10

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

for ((seed=0;seed<seeds;seed+=1))
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
          to_execute+="--env \"$env\" $policy --seed $seed  $device;"
        done
    done
done

if [ -n "$gpu_count" ]; then
    p=$(($1 * gpu_count))
else
    p=$1
fi

echo "$to_execute" | tr ';' '\n' | xargs -P 4 -I {} bash -c "python main_toy_value_show.py {} --save_model"
