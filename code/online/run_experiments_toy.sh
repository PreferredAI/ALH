#!/bin/bash

# Script to reproduce results

to_execute=""

envs=(
	"MultiNormEnv"
	)

share_config="--start_timesteps 25000 --expl_noise 0 --hidden_dim 64 --batch_size 512 --max_timesteps 2000000"
share_not_hard_config="--start_timesteps 25000 --hidden_dim 64  --batch_size 512 -is_not_hard --max_timesteps 2000000"
policies=(
  "--policy memTD32  --hypo_dim 64 $share_config"
  "--policy TD3 $share_config"
  "--policy memTD3  --hypo_dim 64 $share_config"
  "--policy memTD32  --hypo_dim 64 $share_not_hard_config"
  "--policy TD3 $share_not_hard_config"
  "--policy memTD3  --hypo_dim 64 $share_not_hard_config"
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

echo "$to_execute" | tr ';' '\n' | xargs -P $p -I {} bash -c "python main_toy.py {}"
