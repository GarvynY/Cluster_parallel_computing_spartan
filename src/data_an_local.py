#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  @Author: Garvyn-Yuan
  @FIle Name: data_an
  @Contact: 228077gy@163.com
  @Description: a1 - local
    1. But scatter fit uniform data partitioning, and we can not read all data once in the memory for 144GB
  @Date: File created in 11:09-2025/3/28
  @Modified by: 
  @Version: V1.0
"""
import io
import json
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

CHUNK_SIZE = 10  # chunk size for each process

# GBK -> utf-8 forced
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def distribute_data():
    """ rank0 -> read all and split """
    if rank == 0:
        data = []
        with open("data/mastodon-106k.ndjson", "r", encoding="utf-8") as f:
            data = f.readlines()  # read all lines
        print(f"Rank 0 loaded {len(data)} lines")

        # average split for each process
        chunk_size = len(data) // size
        # split data --> List<list>
        chunks = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(size)]

        # if data < chunk size
        if len(data) % size != 0:
            chunks[-1].extend(data[size * chunk_size:])
    else:
        chunks = None

    # scatter the data
    data_chunk = comm.scatter(chunks, root=0)
    return data_chunk


def process_data(data_chunk):
    """ each process processes the data """
    results = []
    for line in data_chunk:
        try:
            data = json.loads(line).get("doc", {})
            value = data.get("sentiment", 0.00)
            user_id = data.get("account", {}).get("id", "")
            user_name = data.get("account", {}).get("username", "")
            create_time = data.get("createdAt", None)
            hour = create_time.split("T")[1][:2] if create_time else None
            results.append([user_id, user_name, hour, value])
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON: {line}")
        except Exception as e:
            print(f"Error processing line: {line}, Error: {e}")
    return results


def gather_results(results):
    """ all together """
    all_results = comm.gather(results, root=0)
    if rank == 0:
        flat_results = [item for sublist in all_results for item in sublist]  # 合并列表
        return flat_results
    return None


# MPI
data_chunk = distribute_data()  # split and send
res = process_data(data_chunk)  # process
all_res = gather_results(res)  # gather

# Rank 0 output
if rank == 0:
    for each in all_res[:10]:  # first 10
        print(each)
