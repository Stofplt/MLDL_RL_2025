#!/bin/bash

# Script to reproduce results

envs=(
	"FetchPush-v1"
	)

for ((i=0;i<1;i+=1))
do 
	for env in ${envs[*]}
	do
		mpirun -np 1 python -m baselines.run_bash_rher \
		--env $env \
		--seed $i \
		--gpu_id=-1
	done
done
