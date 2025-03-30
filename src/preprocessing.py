import ast
import dask.dataframe as dd
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

df = dd.read_json("data/mastodon-106k.ndjson", lines=True, encoding="utf-8")
df = df.repartition(npartitions=size)  # 根据 MPI 进程数重新划分
# 处理每个分区的函数
def process_partition(partition):
    results = []
    for _, row in partition.iterrows():
        try:
            # print(f"Processing row: {row}")

            # 获取 "doc" 字段，确保它是字典
            doc = row.get("doc", {})
            if isinstance(doc, str):
                doc = ast.literal_eval(doc)  # 转换为字典（假设它是一个字符串表示的字典）
            if not isinstance(doc, dict):  # 如果不是字典类型
                # print(f"No valid 'doc' field in row: {row}")
                results.append(["failed", "", "", 0.0])
                continue

            sentiment = doc.get("sentiment", 0.00)
            account = doc.get("account", {})
            userId = account.get("id", "")
            userName = account.get("username", "")
            createTime = doc.get("createdAt", None)
            if createTime is not None:
                hour = createTime.split("T")[1][:2]
            else:
                hour = None

            results.append([userId, userName, hour, sentiment])
        except Exception as e:
            print(f"Error processing row: {e}")
            results.append(["failed", "", "", 0.0])

    return pd.DataFrame(results, columns=["userId", "userName", "hour", "sentiment"])


df = dd.read_json("data/mastodon-106k.ndjson", lines=True, encoding="utf-8")
df_processed = df.map_partitions(process_partition)
df_processed.to_csv("output/processed-*.csv", index=False, single_file=False)
