# Cluster and Cloud Computing

​              			                                                     **Assignment I Report**

## Authors

​									 [**yuan.gao.2@student.unimelb.edu.au**](mailto:yuan.gao.2@student.unimelb.edu.au)                                                            

​								          [**yzhao5067@student.unimelb.edu.au**](mailto:yuan.gao.2@student.unimelb.edu.au)

## **Login and Initialization**

1. use **ssh-key** to login without password, copy public key to the remote host and set ssh config file and alias command locally to login by simply command ‘spartan’.
2. Create shell scripts initial_env.sh and set alias command ‘alias inienv='source /home/ygao3631/initial_env.sh’ in ~/.bashrc ’to **conveniently initialize each time**, The function includes module purge and Load spartan & foss/2022a & Python/3.10.4 & Scipy  

| ![image.png](image.png) | <img src="image%201.png" alt="image.png" style="zoom:150%;" /> |
| ----------------------- | ------------------------------------------------------------ |

## **The slurm scripts for submitting the job**

**The final version:**

```bash
#!/bin/bash
#SBATCH --job-name=mastodon_analysis
#SBATCH --output=mastodon_analysis_%j.out
#SBATCH --error=mastodon_analysis_%j.err
#SBATCH --nodes=[node_amount]
#SBATCH --ntasks-per-node=[core_amount]
#SBATCH --time=04:00:00
#SBATCH --mem=[memory per-core]

mpiexec -n 8 python mastodon_analysis.py
# or sun -n 8 python mastodon_analysis.py
```

For different Jobs, edit ‘--nodes=’; ‘--ntasks-per-node=’  to adjust the different circumstances.

+ **2 nodes ,4 cores for each:** set ‘--nodes=2’; ‘--ntasks-per-node=4’.

+ **1 node, 8 cores for each:** set ‘--nodes=1’; ‘--ntasks-per-node=8’.

+ **1 node ,1 core for each:** set ‘--nodes=1’ ; ‘--ntasks-per-node=1’.

| <img src="/Users/garvyn/Downloads/v3/slurm1.png" alt="slurm1" style="zoom:50%;" /> | <img src="/Users/garvyn/Downloads/v3/slurm3.png" alt="slurm1" style="zoom:50%;" /> | <img src="/Users/garvyn/Downloads/v3/slurm2.png" alt="slurm2" style="zoom:50%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |



## **Approach to build and parallelize the code**

### From stream-based to File-slicing-based MPI design

- **Stream-based MPI design:** The first try uses send and recv to stream data from rank 0 to other processes. While it gives more control over memory and avoids partial line handling, it’s slower because rank 0 reads the entire file alone and becomes the bottleneck for both I/O and communication. This approach runs well on 16m data but take much more time than expected on 144G data: The second version is less efficient might because rank 0 handles all the file reading, which creates a severe I/O bottleneck. Other ranks must wait for data to be sent, causing idle time. Additionally, the frequent use of send and recv introduces high communication overhead, especially when processing large files like 144GB. This serialized workflow limits parallelism and slows down the entire program.
- **File-slicing-based MPI design:** The key optimization is replacing centralized streaming with parallel file reading. Each process reads and processes data independently, and only the final results are gathered, making the whole program much faster and more scalable. The second version lets each process read its own part of the file directly using seek, so all ranks process data in parallel. This greatly improves speed by removing the central bottleneck and reducing communication overhead.
- **Optimization:** Replaced comm.send/recv with direct file slicing using f.seek() ; Let each process handle its own I/O and parsing logic ; Only final aggregation is done on Rank 0 using comm.gather().

```python
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    filename = 'large-144G.ndjson'
    chunk_size = 1024 * 1024 * 100  # 100MB

    hour_sentiment = defaultdict(float)
    user_sentiment = defaultdict(float)

    with open(filename, 'r', encoding='utf-8') as f:
        f.seek(0, 2)
        file_size = f.tell()
        chunk_size = file_size // size
        start = rank * chunk_size
        end = start + chunk_size if rank != size -1 else file_size

        f.seek(start)
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
```

- **Fault tolerance Design:** The code include fault tolerance to deal with the dirty data(e.g. malformed JSON, missing fields) with out crashing

```python
def parse_line(line):
    try:
        data = json.loads(line)  # May raise JSONDecodeError
        doc = data.get('doc', {})  # Prevents KeyError
        created_at = doc.get('createdAt', None)  # Returns None if missing
        sentiment = doc.get('sentiment', None)
        account = doc.get('account', {})
        user_id = account.get('id', None)
        username = account.get('username', None)
        return created_at, sentiment, user_id, username
    except json.JSONDecodeError:
        return None, None, None, None  # Marks invalid records

```

+ **Collect Results in rank 0**

~~~python
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
~~~



## **Performance Analysis**

For 144G dataset, 1 node 8 cores task has the best performance which takes 256 seconds and 2 nodes 4 cores task takes 267 seconds, 1 node 1 core has the worst performance which is 2597 seconds. In each task, the run time on each core is almost the same and the aggregation time is less than 1 second. The reason for 1 node 8 cores job having the best performance is all 8 cores are on the same machine, so there’s no inter-node communication. For 2 nodes 4 cores, (even though it's minimal due to small aggregation time). I/O bandwidth per node might also be lower if shared file system access is not as efficient across nodes. For 1node 1 core, No parallelism at all. One core processes the entire 144GB dataset sequentially, becoming the bottleneck. I/O and CPU are underutilized, leading to extremely long execution time. Finally, the data merging task is lightweight tasks and always completes quickly.

So, The key performance factors are parallelism level, I/O bandwidth, and communication overhead. Maximum performance is achieved when all cores are on the same node and can work in parallel with minimal data transfer delay. The pictures below shows the results for [1-1] [2-4] [1-8] version in order respectively.

| <img src="image%202.png" alt="image.png" style="zoom: 33%;" /> | <img src="image%203.png" alt="image.png" style="zoom: 33%;" /> | <img src="/Users/garvyn/Downloads/v3/image 5.png" alt="image 5" style="zoom: 25%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

+ **Diagram of Results**

<img src="image%204.png" alt="image.png" style="zoom: 67%;" />



### **Amdahl’S Law Application**

Observed speedup ≈ 2597 / 256 ≈ **10.14x ; 2597 / 267** ≈ 9.72x **then** Using Amdahl’s Law to estimate parallel portion P. One node eight cores & Two nodes four cores:

$$
10.14 =  1/((1-p) + (p/8))  → P ≈  0.995
$$

for 2 nodes, Still benefits from high parallelism (~99.5%), but performance loss comes from cross-node communication overhead and shared file system I/O contention.

| **Mode**     | **Time** | **Speed Up** | **Performance**                               |
| ------------ | -------- | ------------ | --------------------------------------------- |
| 1node,1core  | 2597     | 1x           | No parallelism, CPU and I/O are bottlenecks   |
| 1node,8cores | 256      | 10.14x       | Near-optimal scaling, minimal communication   |
| 2node,4cores | 267      | 9.7x         | Effective parallelism, slight inter-node cost |



:heavy_exclamation_mark:**Tips** -- More information about *CI/CD* process can be seen our Github website [https://github.com/GarvynY/Cluster_parallel_computing_spartan.git](ccc.pdf) 
