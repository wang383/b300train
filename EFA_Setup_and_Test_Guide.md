# p6-b300 双机 EFA 配置、测试与训练完整指南

## 环境信息

| 项目 | 节点1（主节点） | 节点2 |
|------|--------------|-------|
| Instance ID | i-0cd02c2c4ac8baa76 | i-05bd8036eecb3ee84 |
| 内网 IP | 172.31.47.177 | 172.31.35.107 |
| AZ | us-west-2a | us-west-2a |
| 安全组 | sg-0957308fc5a969ef0 (efa-nccl-sg) | 同一安全组 |
| ENA 网卡名 | enp71s0 | enp71s0 |
| AWS Profile | myb300 | myb300（通过 SSM 操作） |

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

## 五、SDXL 双机训练

### 训练镜像

```
763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2
```

拉取镜像（两台机器都执行）：
```bash
aws ecr get-login-password --region us-west-2 --profile myb300 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2
```

### 准备目录（两台机器都执行）

```bash
mkdir -p /opt/dlami/nvme/sdxl/{checkpoints,logs}
mkdir -p /opt/dlami/nvme/cache
mkdir -p /opt/dlami/nvme/benchmark/config
```

### accelerate 配置文件

**节点1** `/opt/dlami/nvme/benchmark/config/accelerate_node1.yaml`：
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
machine_rank: 0
num_machines: 2
num_processes: 16
main_process_ip: 172.31.47.177
main_process_port: 29500
mixed_precision: bf16
same_network: true
use_cpu: false
```

**节点2** `/opt/dlami/nvme/benchmark/config/accelerate_node2.yaml`：
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
machine_rank: 1
num_machines: 2
num_processes: 16
main_process_ip: 172.31.47.177
main_process_port: 29500
mixed_precision: bf16
same_network: true
use_cpu: false
```

### 训练内部脚本

**节点1** `/opt/dlami/nvme/benchmark/train_node1_inner.sh`：
```bash
#!/bin/bash
set -e
python -m pip install -q accelerate transformers datasets bitsandbytes
python -m pip install -q git+https://github.com/huggingface/diffusers.git

wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/instruct_pix2pix/train_instruct_pix2pix_sdxl.py \
  -O /workspace/train_instruct_pix2pix_sdxl.py

accelerate launch --config_file /config/accelerate_node1.yaml \
  /workspace/train_instruct_pix2pix_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --dataset_name fusing/instructpix2pix-1000-samples \
  --use_ema --resolution 1024 --train_batch_size 2 \
  --gradient_accumulation_steps 4 --gradient_checkpointing \
  --max_train_steps 500 --learning_rate 5e-6 \
  --lr_scheduler cosine --lr_warmup_steps 50 --mixed_precision bf16 \
  --output_dir /workspace/checkpoints --report_to tensorboard \
  --logging_dir /workspace/logs --checkpointing_steps 100 --seed 42 \
  2>&1 | tee /workspace/train.log
```

**节点2** `/opt/dlami/nvme/benchmark/train_node2_inner.sh`：与节点1相同，`--config_file` 改为 `/config/accelerate_node2.yaml`，日志改为 `/workspace/train_node2.log`。

### 启动训练

获取 infiniband 设备列表（两台机器执行方式相同）：
```bash
IB_DEVICES=$(ls /dev/infiniband/ | grep -v "^by-" | sed 's|^|--device /dev/infiniband/|' | tr '\n' ' ')
```

**先启动节点2**（通过 SSM），**再启动节点1**：

```bash
# 节点1启动命令（节点2同理，替换 node_rank 和脚本路径）
nohup docker run --rm \
  --runtime=nvidia --gpus all \
  --network=host \
  --shm-size=512g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  $IB_DEVICES \
  -v /opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl \
  -v /opt/amazon/efa:/opt/amazon/efa \
  -v /opt/dlami/nvme/sdxl:/workspace \
  -v /opt/dlami/nvme/cache:/root/.cache \
  -v /opt/dlami/nvme/benchmark/config:/config \
  -v /opt/dlami/nvme/benchmark/train_node1_inner.sh:/train.sh \
  -e LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:/usr/local/lib \
  -e FI_EFA_USE_DEVICE_RDMA=1 \
  -e FI_EFA_FORK_SAFE=1 \
  -e NCCL_SOCKET_IFNAME=enp71s0 \
  -e NCCL_IB_HCA=rdmap \
  -e NCCL_DEBUG=WARN \
  -e HF_HOME=/root/.cache/huggingface \
  -e OMP_NUM_THREADS=8 \
  763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2 \
  bash /train.sh \
  > /opt/dlami/nvme/sdxl/train_node1.log 2>&1 &
```

### 训练参数

| 参数 | 值 |
|------|-----|
| 基础模型 | stabilityai/stable-diffusion-xl-base-1.0 |
| 数据集 | fusing/instructpix2pix-1000-samples（1000条） |
| 分辨率 | 1024×1024 |
| 每卡 batch size | 2 |
| Gradient accumulation | 4 |
| 有效 batch size | 2 × 16卡 × 4 = **128** |
| 总训练步数 | 500 |
| 学习率 | 5e-6，cosine scheduler，warmup 50步 |
| 精度 | bf16 |
| Checkpoint 间隔 | 每 100 步 |

### 训练结果（2026-04-20 17:36 启动）

```
[RANK 0] ***** Running training *****
  Num examples = 1000
  Num Epochs = 63
  Total train batch size = 128
  Total optimization steps = 500

step 1:  48.4s/it（warmup）
step 10: 4.66s/it
step 24: 4.11s/it（稳定）
预计总时长：~35 分钟
```

### 监控

```bash
# 训练日志
tail -f /opt/dlami/nvme/sdxl/train_node1.log

# GPU 利用率
nvidia-smi dmon -s u -d 5

# Checkpoint
ls -lh /opt/dlami/nvme/sdxl/checkpoints/
```

### 训练完成后同步到 S3

```bash
aws s3 sync /opt/dlami/nvme/sdxl/checkpoints/ s3://your-bucket/sdxl-checkpoints/ --region us-west-2
```

---

## 六、故障排查

### IBV_WC_REM_ACCESS_ERR（本次遇到并解决）

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

### pip install 不工作

镜像内 pip 有 bug，使用 `python -m pip install` 替代 `pip install`。
