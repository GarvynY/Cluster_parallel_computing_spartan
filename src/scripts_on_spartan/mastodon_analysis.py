from mpi4py import MPI
import json
from datetime import datetime
from collections import defaultdict

def parse_line(line):
    try:
        data = json.loads(line)
        doc = data.get('doc', {})
        created_at = doc.get('createdAt', None)
        sentiment = doc.get('sentiment', None)
        account = doc.get('account', {})
        user_id = account.get('id', None)
        username = account.get('username', None)
        return created_at, sentiment, user_id, username
    except json.JSONDecodeError:
        return None, None, None, None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 每个进程处理文件的一部分
    filename = 'large-144G.ndjson'
    chunk_size = 1024 * 1024 * 100  # 每次读取100MB

    hour_sentiment = defaultdict(float)
    user_sentiment = defaultdict(float)

    with open(filename, 'r', encoding='utf-8') as f:
        # 将文件划分为多个部分，每个进程负责其中一部分
        f.seek(0, 2)  # 移动到文件末尾
        file_size = f.tell()
        chunk_size = file_size // size
        start = rank * chunk_size
        end = start + chunk_size if rank != size -1 else file_size

        f.seek(start)
        # 如果不是第一个进程，跳过当前行以避免部分读取
        if rank != 0:
            f.readline()

        pos = f.tell()
        while pos < end:
            line = f.readline()
            if not line:
                break
            pos = f.tell()
            created_at, sentiment, user_id, username = parse_line(line)
            if created_at and sentiment is not None and user_id and username:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    hour = dt.strftime('%Y-%m-%d %H:00')
                    hour_sentiment[hour] += sentiment
                    user_sentiment[username] += sentiment
                except ValueError:
                    continue

    # 收集所有进程的结果
    all_hour_sentiment = comm.gather(hour_sentiment, root=0)
    all_user_sentiment = comm.gather(user_sentiment, root=0)

    if rank == 0:
        combined_hour = defaultdict(float)
        combined_user = defaultdict(float)

        for hs in all_hour_sentiment:
            for hour, sentiment in hs.items():
                combined_hour[hour] += sentiment

        for us in all_user_sentiment:
            for user, sentiment in us.items():
                combined_user[user] += sentiment

        # 获取5 happiest hours
        happiest_hours = sorted(combined_hour.items(), key=lambda x: x[1], reverse=True)[:5]
        # 获取5 saddest hours
        saddest_hours = sorted(combined_hour.items(), key=lambda x: x[1])[:5]
        # 获取5 happiest users
        happiest_users = sorted(combined_user.items(), key=lambda x: x[1], reverse=True)[:5]
        # 获取5 saddest users
        saddest_users = sorted(combined_user.items(), key=lambda x: x[1])[:5]

        print("5 Happiest Hours:")
        for hour, score in happiest_hours:
            print(f"{hour} with sentiment score {score}")

        print("\n5 Saddest Hours:")
        for hour, score in saddest_hours:
            print(f"{hour} with sentiment score {score}")

        print("\n5 Happiest Users:")
        for user, score in happiest_users:
            print(f"{user} with sentiment score {score}")

        print("\n5 Saddest Users:")
        for user, score in saddest_users:
            print(f"{user} with sentiment score {score}")

if __name__ == "__main__":
    main()