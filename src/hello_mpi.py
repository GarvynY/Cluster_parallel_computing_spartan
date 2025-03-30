#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  @Author: Garvyn-Yuan
  @FIle Name: hello_mpi
  @Contact: 228077gy@163.com
  @Description:
  @Date: File created in 21:40-2025/3/27
  @Modified by: 
  @Version: V1.0
"""
from mpi4py import MPI

comm = MPI.COMM_WORLD # communicator for all process
rank = comm.Get_rank() # get rank of current process
size = comm.Get_size() # get number of processes

print(f"Hello from process {rank} of {size}")

