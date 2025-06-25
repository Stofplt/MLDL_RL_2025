#!/bin/bash

# Script to reproduce results

envs=(
	"FetchPickAndPlace-v1"
	)

export PYTHONPATH=$PYTHONPATH:/home/andrea/Progetto/RHER-main/

for ((i=0;i<5;i+=1))
do 
	for env in ${envs[*]}
	do
		mpirun -np 1 python -m baselines.run_bash_rher \
		--env $env \
		--seed $i \
		--gpu_id=-1
	done
done
