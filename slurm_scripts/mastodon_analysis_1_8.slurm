#!/bin/bash
#SBATCH --job-name=mastodon_analysis
#SBATCH --output=mastodon_analysis_%j.out
#SBATCH --error=mastodon_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=04:00:00
#SBATCH --mem=8G

mpiexec -n 8 python mastodon_analysis.py
