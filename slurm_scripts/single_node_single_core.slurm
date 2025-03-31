#!/bin/bash
#SBATCH --job-name=single_with_single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=mastodon_single_%j.out
#SBATCH --mem=64G

# 运行Python程序
python mpi_parallel_spartan.py small-106k.ndjson --processes 1