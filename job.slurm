#!/bin/bash

### Job name
#SBATCH --job-name=hyperopt_forecast

### Email
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s.w.h.akkerman@students.uu.nl

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
#SBATCH --output /science-nfs-sys/vsm01/users/6397514/Code/Crypto_Forecasting/output_%j.txt
#SBATCH --error /science-nfs-sys/vsm01/users/6397514/Code/Crypto_Forecasting/error.txt

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute:second,
#SBATCH -t 150:00:00

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem 100G

### gpua16 (cs01) or gpua100 (cs02)
#SBATCH --partition=gpua100
###SBATCH

source /science-nfs-sys/vsm01/users/6397514/Code/envs/thesis_env/bin/activate
cd /science-nfs-sys/vsm01/users/6397514/Code/Crypto_Forecasting

python3 src/hyperopt/hyperopt_ray.py