# -*- coding: utf-8 -*-
import json
import sys
import io
from mpi4py import MPI
from datetime import datetime
import os

"""
  @Author: Garvyn-Yuan
  @FIle Name: mpi_parallel_spartan.py
  @Contact: 228077gy@gmail.com
  @Description: cluster computing on spartan ; Sentiment analysis on a large dataset using MPI
  @Date: File created in 15:00-2025/3/29
  @Modified by: Garvyn 3/30
  @Version: V1.0
"""
"""
func:
1. comm.send : blocking when data is copied to os cache or received by recv [blocked until receive process ready] 
2. comm.recv : blocking when receive data (if recv start first, it will wait until match send)
3. [param - tag : match send and recv ]
   [param - dest : destination ]
   [param - source : source of data]
   
mechanism: 
    send wait for recv, but if the data is small, will optimize to put it in the cache and return send, if recv starts
first, it will wait for send . 
    The whole process work as a stream line: rank0-- read,send,receive,gather ; rank x -- receive, process, send back
"""

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# start time
start_time = MPI.Wtime()

# fixed params
CHUNK_SIZE = 4 * 1024 * 1024  # 4MB
DATA_PATH = "medium-16m.ndjson"
# CHUNK_SIZE = 20 * 1024  # 20kb
# DATA_PATH = "data/mastodon-106k.ndjson"
# CHUNK_SIZE = 4 * 1024 * 1024 * 1024  # 4GB
# DATA_PATH = "large-144G.ndjson"

# output stream to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_data_chunk_stream():
    """ rank 0 read ,split and send """
    if rank == 0:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            rank0_buffer = []
            buffer_size = 0

            for line in f:
                rank0_buffer.append(line)
                buffer_size += len(line.encode('utf-8'))  # cal chunk size by cumulating

                # hit the limitation  - > send
                if buffer_size >= CHUNK_SIZE:
                    send_data(rank0_buffer)
                    rank0_buffer = []
                    buffer_size = 0

            # send rest of the data
            if rank0_buffer:
                send_data(rank0_buffer)

        # notify each subprocess when data transfer is done !!! -> so they can continue their work
        for serial_num in range(1, size):
            comm.send(None, dest=serial_num, tag=1)


def send_data(data_chunk):
    """ send data to workers """
    num_workers = size - 1
    chunk_size = len(data_chunk) // num_workers

    for serial_num in range(1, size):
        # start pointer
        start = (serial_num - 1) * chunk_size
        # end pointer
        end = len(data_chunk) if serial_num == num_workers else serial_num * chunk_size
        cur_data = data_chunk[start:end]
        print(f"Rank 0 sending {len(cur_data)} lines to Rank {serial_num}")
        comm.send(cur_data, dest=serial_num, tag=1)


def process_and_aggregate():
    """ subprocess fetch data, process and send back """
    while True:
        # print(f"Rank {rank} receiving data......")
        data_chunk = comm.recv(source=0, tag=1)
        if data_chunk is None:
            print(f"Rank {rank} received termination signal.")
            break  # done

        # print(f"Rank {rank} received {len(data_chunk)} entries")
        # comm.barrier()

        user_sentiments = {}  # user sentiment
        hour_sentiments = {}  # time sentiment（hour）

        for line in data_chunk:
            try:
                # set default value to fill the empty position
                data = json.loads(line).get("doc", {})
                sentiment = data.get("sentiment", 0.00)
                user_id = data.get("account", {}).get("id", "")
                username = data.get("account", {}).get("username", "")
                created_at = data.get("createdAt", None)

                if not user_id or not username or sentiment is None:
                    continue

                # aggregate by (user_id, username)
                user_key = (user_id, username)
                if user_key not in user_sentiments:
                    user_sentiments[user_key] = 0.0
                user_sentiments[user_key] += sentiment

                # cumulate by hour sentiment
                if created_at:
                    hour_key = created_at[:13]  # eg: "2023-11-23T15"
                    if hour_key not in hour_sentiments:
                        hour_sentiments[hour_key] = 0.0
                    hour_sentiments[hour_key] += sentiment

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

        # send results to rank 0
        comm.send((user_sentiments, hour_sentiments), dest=0, tag=2)


def gather_results():
    """ rank 0 collects all results and aggregate """
    final_user_sentiments = {}
    final_hour_sentiments = {}

    for serial_num in range(1, size):
        user_data, hour_data = comm.recv(source=serial_num, tag=2)
        # print(user_data)
        # print(hour_data)

        # aggregate people data
        for (user_id, username), sentiment in user_data.items():
            user_key = (user_id, username)
            if user_key not in final_user_sentiments:
                final_user_sentiments[user_key] = 0.0
            final_user_sentiments[user_key] += sentiment

        # aggregate hour data
        for hour, sentiment in hour_data.items():
            if hour not in final_hour_sentiments:
                final_hour_sentiments[hour] = 0.0
            final_hour_sentiments[hour] += sentiment

    # happiest/saddest people -- sort by score（the second term）
    sorted_users = sorted(final_user_sentiments.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_users)
    happiest_users = sorted_users[:5]
    saddest_users = sorted_users[-5:]

    # happiest/saddest hours -- sort by score（the second term）
    sorted_hours = sorted(final_hour_sentiments.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_hours)
    happiest_hours = sorted_hours[:5]
    saddest_hours = sorted_hours[-5:]

    # 生成文件名: large-144G.ndjson_2025-03-31_17-30-00_results.txt
    output_filename = f"{os.path.basename(DATA_PATH)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_results.txt"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("Top 5 Happiest Users:\n")
        for (user_id, username), score in happiest_users:
            f.write(f"User ID: {user_id}, Username: {username}, Sentiment Score: {score:.2f}\n")

        f.write("\nTop 5 Saddest Users:\n")
        for (user_id, username), score in saddest_users:
            f.write(f"User ID: {user_id}, Username: {username}, Sentiment Score: {score:.2f}\n")

        f.write("\nTop 5 Happiest Hours:\n")
        for hour, score in happiest_hours:
            f.write(f"{hour} - Sentiment Score: {score:.2f}\n")

        f.write("\nTop 5 Saddest Hours:\n")
        for hour, score in saddest_hours:
            f.write(f"{hour} - Sentiment Score: {score:.2f}\n")


    print(f"Results saved to {output_filename}")
    # # res
    # print("\nTop 5 Happiest Users:")
    # for (user_id, username), score in happiest_users:
    #     print(f"User ID: {user_id}, Username: {username}, Sentiment Score: {score:.2f}")
    #
    # print("\nTop 5 Saddest Users:")
    # for (user_id, username), score in saddest_users:
    #     print(f"User ID: {user_id}, Username: {username}, Sentiment Score: {score:.2f}")
    #
    # print("\nTop 5 Happiest Hours:")
    # for hour, score in happiest_hours:
    #     print(f"{hour} - Sentiment Score: {score:.2f}")
    #
    # print("\nTop 5 Saddest Hours:")
    # for hour, score in saddest_hours:
    #     print(f"{hour} - Sentiment Score: {score:.2f}")


# # version 1 -- run test

# if rank == 0:
#     read_data_in_chunks()
#     gather_results()
# else:
#     # can not be initialized as process_and_aggregate([]) here because when each subprocess started , some may get [].
#     process_and_aggregate()
#
# # process synchronization
# comm.barrier()
# # end time
# end_time = MPI.Wtime()
# if rank == 0:
#     print(f"Total execution time: {end_time - start_time:.2f} seconds")


# version 2: main process and time stat
if size == 1:
    # 单进程模式
    print("Running in single-process mode.")
    start_time = MPI.Wtime()

    # 读取数据并直接处理
    read_start = MPI.Wtime()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data_chunk = f.readlines()
    read_end = MPI.Wtime()
    print(f"Data reading time: {read_end - read_start:.2f} seconds")

    # 处理数据
    processing_start = MPI.Wtime()
    user_sentiments = {}
    hour_sentiments = {}

    for line in data_chunk:
        try:
            data = json.loads(line).get("doc", {})
            sentiment = data.get("sentiment", 0.00)
            user_id = data.get("account", {}).get("id", "")
            username = data.get("account", {}).get("username", "")
            created_at = data.get("createdAt", None)

            if not user_id or not username or sentiment is None:
                continue

            user_key = (user_id, username)
            user_sentiments[user_key] = user_sentiments.get(user_key, 0.0) + sentiment

            if created_at:
                hour_key = created_at[:13]  # eg: "2023-11-23T15"
                hour_sentiments[hour_key] = hour_sentiments.get(hour_key, 0.0) + sentiment
        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"Error processing line: {e}")
            continue

    processing_end = MPI.Wtime()
    print(f"Processing time: {processing_end - processing_start:.2f} seconds")

    # 结果排序
    gather_start = MPI.Wtime()
    sorted_users = sorted(user_sentiments.items(), key=lambda x: x[1], reverse=True)
    happiest_users = sorted_users[:5]
    saddest_users = sorted_users[-5:]

    sorted_hours = sorted(hour_sentiments.items(), key=lambda x: x[1], reverse=True)
    happiest_hours = sorted_hours[:5]
    saddest_hours = sorted_hours[-5:]

    gather_end = MPI.Wtime()
    print(f"Gathering results time: {gather_end - gather_start:.2f} seconds")

    # 输出到文件
    output_filename = f"{os.path.basename(DATA_PATH)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_results.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("Top 5 Happiest Users:\n")
        for (user_id, username), score in happiest_users:
            f.write(f"User ID: {user_id}, Username: {username}, Sentiment Score: {score:.2f}\n")

        f.write("\nTop 5 Saddest Users:\n")
        for (user_id, username), score in saddest_users:
            f.write(f"User ID: {user_id}, Username: {username}, Sentiment Score: {score:.2f}\n")

        f.write("\nTop 5 Happiest Hours:\n")
        for hour, score in happiest_hours:
            f.write(f"{hour} - Sentiment Score: {score:.2f}\n")

        f.write("\nTop 5 Saddest Hours:\n")
        for hour, score in saddest_hours:
            f.write(f"{hour} - Sentiment Score: {score:.2f}\n")

    print(f"Results saved to {output_filename}")

    end_time = MPI.Wtime()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

else:
    # MPI 并行模式
    if rank == 0:
        read_start = MPI.Wtime()
        load_data_chunk_stream()
        read_end = MPI.Wtime()
        print(f"Data reading and distribution time: {read_end - read_start:.2f} seconds")

    processing_start = MPI.Wtime()
    if rank != 0:
        process_and_aggregate()
    processing_end = MPI.Wtime()
    if rank != 0:
        print(f"Rank {rank} processing time: {processing_end - processing_start:.2f} seconds")

    gather_start = MPI.Wtime()
    if rank == 0:
        gather_results()
    gather_end = MPI.Wtime()
    if rank == 0:
        print(f"Results gathering time: {gather_end - gather_start:.2f} seconds")

    comm.barrier()
    MPI.Finalize()
    end_time = MPI.Wtime()
    if rank == 0:
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

