#!/bin/bash
#SBATCH --job-name=mastodon_analysis
#SBATCH --output=mastodon_analysis_11%j.out
#SBATCH --error=mastodon_analysis_11%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=8G

mpiexec -n 1 python mastodon_analysis.py
