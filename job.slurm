#!/bin/bash

### Job name
#SBATCH --job-name=stress_test_all

### Email
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s.w.h.akkerman@students.uu.nl

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
#SBATCH --output /nethome/6397514/Scratch/code/timeseries-volatility-influence/output_%j.txt
#SBATCH --error /nethome/6397514/Scratch/code/timeseries-volatility-influence/error.txt

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute:second,
#SBATCH -t 360:00:00
# 15 days

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem 100G

### gpua16 (cs01) or gpua100 (cs02)
#SBATCH --partition=gpua100
###SBATCH

source /nethome/6397514/Scratch/venvs/thesis_env/bin/activate
cd /nethome/6397514/Scratch/code/timeseries-volatility-influence

srun python3 src/main.py