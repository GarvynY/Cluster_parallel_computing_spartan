#!/bin/bash
#SBATCH --job-name=2_4v       # 任务名称
#SBATCH --nodes=2                  # 申请 2 个计算节点
#SBATCH --ntasks-per-node=4        # 每个节点运行 4 个 MPI 任务 (共 8 进程)
#SBATCH --cpus-per-task=1          # 每个 MPI 任务使用 1 个 CPU 核
#SBATCH --time=01:00:00            # 运行时间上限 1 小时
#SBATCH --mem=32G                  # 申请 32GB 内存
#SBATCH --output=mpi_%j.log     # 标准输出日志
#SBATCH --error=mpi_%j.log       # 标准错误日志

srun python mpi_parallel_spartan_16m.py