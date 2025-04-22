#!/bin/bash
#SBATCH --job-name=single_8     # 任务名称
#SBATCH --nodes=1                    # 申请1个节点
#SBATCH --ntasks-per-node=8           # 每个节点上 8 个任务 (共8进程)
#SBATCH --cpus-per-task=1             # 每个 MPI 进程使用 1 个 CPU 核
#SBATCH --time=01:00:00               # 最大运行时间 1 小时
#SBATCH --mem=32G                    # 申请 32GB 内存
#SBATCH --output=mpi_%j.log        # 标准输出日志
#SBATCH --error=mpi_%j.log          # 错误日志

srun python mpi_parallel_spartan_16m.py