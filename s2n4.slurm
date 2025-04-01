#!/bin/bash
#SBATCH --job-name=mastodon_analysis
#SBATCH --output=mastodon_analysis.out
#SBATCH --error=mastodon_analysis.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --mem=8G

mpiexec -n 8 python mastodon_analysis.py
