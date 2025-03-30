#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  @Author: Garvyn-Yuan
  @FIle Name: mpi_parellel_spartan.py
  @Contact: 228077gy@163.com
  @Description:
  @Date: File created in 15:00-2025/3/30
  @Modified by: 
  @Version: V1.0
"""

"""
  @Author: Garvyn-Yuan
  @File Name: mpi_parellel_spartan.py
  @Contact: 228077gy@gamil.com
  @Description: Sentiment analysis on a large dataset using MPI
  @Date: 2025-03-28
  @Version: V1.0
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import io
from mpi4py import MPI

# MPI 初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 设定读取的文件块大小
CHUNK_SIZE = 20 * 1024
DATA_PATH = "data/mastodon-106k.ndjson"

# 确保 Python 输出 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def read_data_in_chunks():
    """ rank 0 逐块读取数据，并分发到各进程 """
    if rank == 0:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            buffer = []
            buffer_size = 0

            for line in f:
                buffer.append(line)
                buffer_size += len(line.encode('utf-8'))

                # 达到 CHUNK_SIZE 后，发送数据
                if buffer_size >= CHUNK_SIZE:
                    send_data(buffer)
                    buffer = []
                    buffer_size = 0

            # 发送剩余数据
            if buffer:
                send_data(buffer)

        # 通知所有进程数据结束
        for _ in range(1, size):
            comm.send(None, dest=_, tag=1)


def send_data(data_chunk):
    """ rank 0 将数据块分发给各个进程 """
    num_workers = size - 1
    chunk_size = len(data_chunk) // num_workers

    for i in range(1, size):
        start = (i - 1) * chunk_size
        end = len(data_chunk) if i == num_workers else i * chunk_size
        comm.send(data_chunk[start:end], dest=i, tag=1)
        print(f"Sent {len(data_chunk[start:end])} lines to Rank {i}")  # 调试信息


def process_and_aggregate():
    """ 子进程处理数据并聚合 """
    while True:
        data_chunk = comm.recv(source=0, tag=1)
        if data_chunk is None:
            break  # 结束信号

        print(f"Rank {rank} received {len(data_chunk)} entries")  # 调试信息

        user_sentiments = {}  # 存储用户 sentiment
        hour_sentiments = {}  # 存储时间 sentiment（小时）

        for line in data_chunk:
            try:
                data = json.loads(line).get("doc", {})
                sentiment = data.get("sentiment", 0.00)
                user_id = data.get("account", {}).get("id", "")
                username = data.get("account", {}).get("username", "")
                created_at = data.get("createdAt", None)

                if not user_id or sentiment is None:
                    continue

                # 按 (user_id, username) 累加 sentiment
                user_key = (user_id, username)
                if user_key not in user_sentiments:
                    user_sentiments[user_key] = 0.0
                user_sentiments[user_key] += sentiment

                # 按小时累加 sentiment
                if created_at:
                    hour_key = created_at[:13]  # 取到小时 "2023-11-23T15"
                    if hour_key not in hour_sentiments:
                        hour_sentiments[hour_key] = 0.0
                    hour_sentiments[hour_key] += sentiment

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

        # 发送处理结果回 rank 0
        comm.send((user_sentiments, hour_sentiments), dest=0, tag=2)


def gather_results():
    """ rank 0 收集所有进程的结果，并计算最终结果 """
    final_user_sentiments = {}
    final_hour_sentiments = {}

    for _ in range(1, size):
        user_data, hour_data = comm.recv(source=_, tag=2)
        # print(user_data)
        # print(hour_data)

        # 合并用户 sentiment 结果
        for (user_id, username), sentiment in user_data.items():
            user_key = (user_id, username)
            if user_key not in final_user_sentiments:
                final_user_sentiments[user_key] = 0.0
            final_user_sentiments[user_key] += sentiment

        # 合并小时 sentiment 结果
        for hour, sentiment in hour_data.items():
            if hour not in final_hour_sentiments:
                final_hour_sentiments[hour] = 0.0
            final_hour_sentiments[hour] += sentiment

    # 计算最快乐/最不快乐的用户
    sorted_users = sorted(final_user_sentiments.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_users)
    happiest_users = sorted_users[:5]
    saddest_users = sorted_users[-5:]

    # 计算最快乐/最不快乐的小时
    sorted_hours = sorted(final_hour_sentiments.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_hours)
    happiest_hours = sorted_hours[:5]
    saddest_hours = sorted_hours[-5:]

    # 输出结果
    print("\nTop 5 Happiest Users:")
    for (user_id, username), score in happiest_users:
        print(f"User ID: {user_id}, Username: {username}, Sentiment Score: {score:.2f}")

    print("\nTop 5 Saddest Users:")
    for (user_id, username), score in saddest_users:
        print(f"User ID: {user_id}, Username: {username}, Sentiment Score: {score:.2f}")

    print("\nTop 5 Happiest Hours:")
    for hour, score in happiest_hours:
        print(f"{hour} - Sentiment Score: {score:.2f}")

    print("\nTop 5 Saddest Hours:")
    for hour, score in saddest_hours:
        print(f"{hour} - Sentiment Score: {score:.2f}")


# 运行程序
if rank == 0:
    read_data_in_chunks()
    gather_results()
else:
    process_and_aggregate()


