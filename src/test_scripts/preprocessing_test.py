#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  @Author: Garvyn-Yuan
  @FIle Name: preprocessing
  @Contact: 228077gy@163.com
  @Description:
  @Date: File created in 21:27-2025/3/27
  @Modified by: 
  @Version: V1.0
"""
from mpi4py import MPI
import json

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# read ndjson file from rank 0 process
if rank == 0:
    with open("data/mastodon-106k.ndjson", "r", encoding="utf-8") as f:
        lines = f.readlines()
else:
    lines = None

# broadcast to all processes
lines = comm.bcast(lines, root=0)

# each process deal with data
local_data = lines[rank::size]
sentiments = []

for line in local_data:
    try:
        doc = json.loads(line)
        if "sentiment" in doc:
            sentiments.append(doc["sentiment"])
    except json.JSONDecodeError:
        continue  # 跳过错误数据

# gather
avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
global_avg = comm.reduce(avg_sentiment, op=MPI.SUM, root=0)

if rank == 0:
    print("sentiment score mean:", global_avg / size)
