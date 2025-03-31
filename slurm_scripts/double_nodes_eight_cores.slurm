#!/bin/bash
#SBATCH --job-name=mastodon_two_nodes
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=mastodon_two_nodes_%j.out
#SBATCH --mem=64G

# 加载必要的模块
module load Python/3.8.2

# 运行Python程序
python mastodon_analysis.py /data/gpfs/projects/COMP90024/mastodon-144Gb.ndjson --processes 8