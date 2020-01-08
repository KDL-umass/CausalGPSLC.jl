#!/bin/bash
#
#SBATCH --job-name=GPROC_EXP
#SBATCH --output=cluster_outputs/res_%j.txt  # output file
#SBATCH -e cluster_outputs/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --array=1-40

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

echo "Start running experiments"

julia run_inference_IHDB.jl $SLURM_ARRAY_TASK_ID