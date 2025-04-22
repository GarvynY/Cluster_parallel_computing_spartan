#!/bin/bash
#SBATCH --job-name=s_s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=single_%j.out
#SBATCH --mem=64G

# 运行Python程序
python mpi_parallel_spartan_single.py
