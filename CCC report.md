# CCC report

**最终提交与交付**

你需要撰写一份简短的报告（不超过4页！），说明如何调用应用程序，包括提交作业到SPARTAN的脚本、并行化代码的方法，以及在不同节点和核心数量下的性能差异。报告中应包含上述实际结果表格，以及一张图表（如条形图）显示在1节点1核心、1节点8核心和2节点8核心配置下的执行时间。你还应将结果与阿姆达尔定律（Amdahl’s law）联系起来，描述潜在的性能变化。

### 程序调用和运行过程

### slurm脚本

### 代码方法和构建过程：

### **Data Partitioning Strategy**

### **Objective**

Distribute the large NDJSON file evenly across multiple MPI processes to prevent memory overload on a single process.

### **Implementation**

- **Byte-Offset Partitioning**: Each process handles a contiguous byte block (not line-based, to avoid splitting JSON records).
- **Boundary Alignment**: Non-root processes skip the first line to ensure reading starts from a complete JSON record.

### **Corresponding Code**

```python
with open(filename, 'r', encoding='utf-8') as f:
    # Calculate total file size and byte ranges for each process
    f.seek(0, 2)  # Move to end of file
    file_size = f.tell()
    chunk_size = file_size // size  # Divide bytes equally
    start = rank * chunk_size
    end = start + chunk_size if rank != size -1 else file_size  # Last process handles remaining bytes

    f.seek(start)
    # Non-root processes skip potentially truncated lines
    if rank != 0:
        f.readline()  # Key: Align to the start of a complete line

```

### **Why It Works**

- Prevents JSON record truncation (`readline()` ensures starting from a line boundary).
- Large chunk sizes (100MB-level) minimize frequent I/O operations.

---

### **2. Parallel Processing and Local Aggregation**

### **Objective**

Each process independently calculates sentiment scores for its assigned time windows and users, reducing subsequent communication overhead.

### **Implementation**

- **Local Dictionary Storage**: Uses `defaultdict` to accumulate results per process.
- **Time Standardization**: Converts timestamps to `YYYY-MM-DD HH:00` format for consistent merging.

### **Corresponding Code**

```python
hour_sentiment = defaultdict(float)  # Time-window sentiment scores
user_sentiment = defaultdict(float)  # User sentiment scores

while pos < end:
    line = f.readline()
    if not line:
        break
    pos = f.tell()
    created_at, sentiment, user_id, username = parse_line(line)
    if all([created_at, sentiment is not None, user_id, username]):
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            hour = dt.strftime('%Y-%m-%d %H:00')  # Standardize to hourly granularity
            hour_sentiment[hour] += sentiment
            user_sentiment[username] += sentiment
        except ValueError:
            continue  # Skip invalid timestamps

```

### **Why It Works**

- Local aggregation drastically reduces data transfer volume (raw JSON → key-value pairs).
- Time standardization ensures correct merging of identical windows across processes.

---

### **3. Result Aggregation and Global Ranking**

### **Objective**

Combine partial results from all processes to generate global rankings.

### **Implementation**

- **MPI Gather Operation**: The root process (Rank 0) collects dictionaries from all worker processes.
- **Two-Phase Merging**: First merges time-window scores, then user scores.

### **Corresponding Code**

```python
# Gather partial results from all processes
all_hour_sentiment = comm.gather(hour_sentiment, root=0)
all_user_sentiment = comm.gather(user_sentiment, root=0)

if rank == 0:
    # Merge time-window scores
    combined_hour = defaultdict(float)
    for hs in all_hour_sentiment:  # hs is one process's hour_sentiment
        for hour, sentiment in hs.items():
            combined_hour[hour] += sentiment

    # Merge user scores (similar to time-window merging)
    combined_user = defaultdict(float)
    for us in all_user_sentiment:
        for user, sentiment in us.items():
            combined_user[user] += sentiment

    # Global ranking (Top 5)
    happiest_hours = sorted(combined_hour.items(), key=lambda x: x[1], reverse=True)[:5]
    saddest_hours = sorted(combined_hour.items(), key=lambda x: x[1])[:5]
    happiest_users = sorted(combined_user.items(), key=lambda x: x[1], reverse=True)[:5]
    saddest_users = sorted(combined_user.items(), key=lambda x: x[1])[:5]

```

### **Why It Works**

- `comm.gather` centralizes distributed results to the root process.
- The root processes only handles merged dictionaries, not raw data (decouples computation and communication).

---

### **4. Fault Tolerance Design**

### **Objective**

Handle dirty data (e.g., malformed JSON, missing fields) without crashing.

### **Key Code**

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

### **Logic**

- **Invalid Record Skipping**: When `parse_line` returns `None`, the main loop filters them via `if all([...])`.
- **Timestamp Resilience**: `try-except` catches illegal time formats (e.g., non-ISO strings).

---

### **5. Performance Optimization Notes**

### **I/O Bottleneck**

- **Issue**: Concurrent file reads by multiple processes may cause contention.
- **Suggestion**:
    
    ```python
    # Use memory-mapped files (requires binary mode)
    with open(filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # Read 'mm' object by byte offsets
    
    ```
    

### **Load Balancing**

- **Issue**: Uneven sentiment score distribution may lead to workload imbalance.
- **Suggestion**:
    - Dynamically allocate data chunks (e.g., root process distributes rows on-demand).
    - Use `MPI_Scatterv` instead of fixed partitioning.

---

### **Conclusion**

Our parallelization strategy effectively leverages HPC resources through **data partitioning + local aggregation + global merging**. Key strengths:

1. **Coarse-Grained Partitioning**: Reduces inter-process communication.
2. **Dictionary Merging**: Lowers memory pressure on the root process.
3. **Fault Tolerance**: Ensures long-running stability.

For further optimization, consider the I/O and load-balancing suggestions above.

### 性能分析

amdahl’S law+三种模式下的性能差距