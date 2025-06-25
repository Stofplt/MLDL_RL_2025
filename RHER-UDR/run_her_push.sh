#!/bin/bash

# Script to reproduce results

envs=(
	"FetchPush-v1"
	)

# Explicitly add the project root to PYTHONPATH
# This tells Python to look for packages here first.
export PYTHONPATH=$PYTHONPATH:/home/andrea/Progetto/RHER-main/

for ((i=0;i<1;i+=1))
do
	for env in ${envs[*]}
	do
		mpirun -np 1 python -m baselines.run_bash_her \
		--env $env \
		--seed $i \
		--gpu_id=-1
	done
done