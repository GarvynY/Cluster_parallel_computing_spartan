#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  @Author: Garvyn-Yuan
  @FIle Name: data_an
  @Contact: 228077gy@163.com
  @Description: a1
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

CHUNK_SIZE = 10  # 每个进程的数据块大小

# 解决 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def distribute_data():
    """ 仅 rank 0 读取数据并分块 """
    if rank == 0:
        data = []
        with open("data/mastodon-106k.ndjson", "r", encoding="utf-8") as f:
            data = f.readlines()  # 直接读取所有行
        print(f"Rank 0 loaded {len(data)} lines")

        # 计算每个进程应分配的行数
        chunk_size = len(data) // size
        chunks = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(size)]

        # 可能有剩余数据，放到最后一个进程
        if len(data) % size != 0:
            chunks[-1].extend(data[size * chunk_size:])
    else:
        chunks = None

    # 分发数据
    data_chunk = comm.scatter(chunks, root=0)
    return data_chunk


def process_data(data_chunk):
    """ 处理每个进程接收到的数据 """
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
            print(f"Skipping invalid JSON: {line}")  # 仅调试用
        except Exception as e:
            print(f"Error processing line: {line}, Error: {e}")
    return results


def gather_results(results):
    """ 汇总所有进程的结果 """
    all_results = comm.gather(results, root=0)
    if rank == 0:
        flat_results = [item for sublist in all_results for item in sublist]  # 合并列表
        return flat_results
    return None


# 运行 MPI 任务
data_chunk = distribute_data()  # 分发数据
res = process_data(data_chunk)  # 处理数据
all_res = gather_results(res)  # 汇总数据

# Rank 0 输出最终结果
if rank == 0:
    for each in all_res[:10]:  # 仅输出前 10 条
        print(each)
