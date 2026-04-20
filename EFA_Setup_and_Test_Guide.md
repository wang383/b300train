# p6-b300 双机 EFA 配置与测试指南

## 环境信息

| 项目 | 节点1（主节点） | 节点2 |
|------|--------------|-------|
| Instance ID | i-0cd02c2c4ac8baa76 | i-05bd8036eecb3ee84 |
| 内网 IP | 172.31.47.177 | 172.31.35.107 |
| AZ | us-west-2a | us-west-2a |
| 安全组 | sg-0957308fc5a969ef0 (efa-nccl-sg) | 同一安全组 |
| ENA 网卡名 | enp71s0 | enp71s0 |

---

## 一、EFA 验证

两台机器启动后，验证 EFA 是否正常加载：

```bash
# 验证内核模块
lsmod | grep efa
# 期望输出：efa  131072  0

# 验证 EFA 设备（应看到 16 个 rdmap* 设备）
/opt/amazon/efa/bin/fi_info -p efa | grep domain

# 验证 uverbs 设备节点
ls /dev/infiniband/
```

### p6-b300 的网络设备说明

p6-b300 上存在两类 RDMA 设备，必须区分：

| 设备名 | 类型 | 用途 |
|--------|------|------|
| `rdmap86s0` ~ `rdmap192s0`（共16个） | EFA（vendor 0x1d0f） | **跨节点通信，应使用** |
| `ibp198s0f0`、`ibp199s0f0` | Mellanox InfiniBand（vendor 0x02c9） | NVSwitch 内部 fabric，**不可用于跨节点** |

用 `ibv_devinfo` 可以确认设备类型：
```bash
/opt/amazon/efa/bin/ibv_devinfo | grep -E "hca_id|vendor_id|transport"
```

---

## 二、安全组配置

两台机器使用同一安全组 `efa-nccl-sg`，已有规则：**同安全组内所有流量互通**（IpProtocol=-1）。

无需额外配置。验证方式：
```bash
aws ec2 describe-security-groups --group-ids sg-0957308fc5a969ef0 \
  --region us-west-2 --profile myb300 \
  --query 'SecurityGroups[0].IpPermissions'
```

---

## 三、关键 NCCL 环境变量

### 问题根因

NCCL 默认会扫描所有 IB/RDMA 设备，会选中 `ibp198s0f0`（Mellanox IB），该设备是 NVSwitch 内部 fabric，active_mtu 只有 512，用于跨节点通信时报错：

```
NET/IB: Got completion from peer with status=IBV_WC_REM_ACCESS_ERR(10)
NET/IB: ibp198s0f0:1 async fatal event on QP: local access violation work queue error
```

### 解决方案

加 `NCCL_IB_HCA=rdmap` 强制 NCCL 只使用名称以 `rdmap` 开头的 EFA 设备：

```bash
-e NCCL_IB_HCA=rdmap
```

### 完整 NCCL/EFA 环境变量

```bash
-e FI_EFA_USE_DEVICE_RDMA=1      # 启用 EFA RDMA
-e FI_EFA_FORK_SAFE=1            # 多进程 fork 安全
-e NCCL_SOCKET_IFNAME=enp71s0    # NCCL bootstrap 使用 ENA 网卡
-e NCCL_IB_HCA=rdmap             # 关键：只使用 EFA 设备，排除 ibp* IB 网卡
-e NCCL_DEBUG=WARN               # 日志级别
```

---

## 四、EFA 通信测试

### 测试脚本

```python
# /tmp/efa_test.py
import os, torch, torch.distributed as dist, time

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

sizes = [1*1024*1024, 64*1024*1024, 512*1024*1024]
for sz in sizes:
    t = torch.ones(sz // 4, dtype=torch.float32, device=f"cuda:{local_rank}")
    dist.barrier()
    t0 = time.time()
    for _ in range(10):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    bw = sz * 10 * 2 / elapsed / 1e9
    if rank == 0:
        print(f"size={sz//1024//1024}MB  time={elapsed/10*1000:.1f}ms  algbw={bw:.1f}GB/s")

if rank == 0:
    print("EFA all_reduce test PASSED")
dist.destroy_process_group()
```

### 启动测试

获取所有 infiniband 设备参数：
```bash
IB_DEVICES=$(ls /dev/infiniband/ | grep -v "^by-" | sed 's|^|--device /dev/infiniband/|' | tr '\n' ' ')
```

**节点2先启动（rank 1）：**
```bash
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=64g --ulimit memlock=-1 \
  $IB_DEVICES \
  -v /opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl \
  -v /opt/amazon/efa:/opt/amazon/efa \
  -v /tmp/efa_test.py:/efa_test.py \
  -e LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:/usr/local/lib \
  -e FI_EFA_USE_DEVICE_RDMA=1 -e FI_EFA_FORK_SAFE=1 \
  -e NCCL_SOCKET_IFNAME=enp71s0 -e NCCL_DEBUG=WARN -e NCCL_IB_HCA=rdmap \
  763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2 \
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=172.31.47.177 --master_port=29501 /efa_test.py
```

**节点1随后启动（rank 0）：**
```bash
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=64g --ulimit memlock=-1 \
  $IB_DEVICES \
  -v /opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl \
  -v /opt/amazon/efa:/opt/amazon/efa \
  -v /tmp/efa_test.py:/efa_test.py \
  -e LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:/usr/local/lib \
  -e FI_EFA_USE_DEVICE_RDMA=1 -e FI_EFA_FORK_SAFE=1 \
  -e NCCL_SOCKET_IFNAME=enp71s0 -e NCCL_DEBUG=WARN -e NCCL_IB_HCA=rdmap \
  763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2 \
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=172.31.47.177 --master_port=29501 /efa_test.py
```

### 测试结果（2026-04-20）

```
size=1MB    time=5.9ms   algbw=0.4 GB/s
size=64MB   time=9.0ms   algbw=14.8 GB/s
size=512MB  time=61.6ms  algbw=17.4 GB/s
EFA all_reduce test PASSED
```

---

## 五、故障排查

### IBV_WC_REM_ACCESS_ERR

```
NET/IB: Got completion from peer with status=IBV_WC_REM_ACCESS_ERR(10)
NET/IB: ibp198s0f0:1 async fatal event on QP: local access violation work queue error
```

**原因：** NCCL 选择了 `ibp198s0f0`（NVSwitch IB），不支持跨节点 RDMA。  
**修复：** 添加 `-e NCCL_IB_HCA=rdmap`

### fi_info 命令找不到

```bash
# 使用完整路径
/opt/amazon/efa/bin/fi_info -p efa
```

### NCCL rendezvous timeout

节点2必须在节点1启动后 15 分钟内完成连接。建议先启动节点2，再启动节点1。
