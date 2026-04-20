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
| 宿主机 OS | Ubuntu 24.04，glibc 2.39 | 同左 |

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

无需额外配置。

---

## 三、关键 NCCL 环境变量

### 问题1：NCCL 选错网卡（IBV_WC_REM_ACCESS_ERR）

**现象：**
```
NET/IB: Got completion from peer with status=IBV_WC_REM_ACCESS_ERR(10)
NET/IB: ibp198s0f0:1 async fatal event on QP: local access violation work queue error
```

**原因：** NCCL 默认扫描所有 IB/RDMA 设备，选中了 `ibp198s0f0`（NVSwitch 内部 IB），该设备 active_mtu 只有 512，不能用于跨节点通信。

**修复：** 加 `NCCL_IB_HCA=rdmap` 强制只使用 EFA 设备：
```bash
-e NCCL_IB_HCA=rdmap
```

### 完整 NCCL/EFA 环境变量

```bash
-e FI_EFA_USE_DEVICE_RDMA=1      # 启用 EFA RDMA
-e FI_EFA_FORK_SAFE=1            # 多进程 fork 安全
-e NCCL_SOCKET_IFNAME=enp71s0    # NCCL bootstrap 使用 ENA 网卡
-e NCCL_IB_HCA=rdmap             # 只使用 EFA 设备，排除 ibp* IB 网卡
-e NCCL_DEBUG=WARN               # 日志级别
-e NCCL_P2P_DISABLE=1            # 禁用 P2P（避免 GDRCopy 初始化卡住）
```

---

## 四、Docker 镜像问题与解决

### 问题2：ofi-nccl 插件加载失败（glibc 版本不兼容）

**现象（NCCL_DEBUG=INFO 可见）：**
```
NET/Plugin: libnccl-net.so: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
NET/IB : No device found.
Failed to initialize NET plugin IB
Initialized NET plugin Socket
Channel 00/0 : 1[0] -> 0[0] [receive] via NET/Socket/0
```

**根因：**
- 宿主机：Ubuntu 24.04，glibc **2.39**
- 容器镜像（pytorch-training:2.10.0...ubuntu22.04）：Ubuntu 22.04，glibc **2.35**
- 宿主机上的 ofi-nccl（`/opt/amazon/ofi-nccl`）是为 Ubuntu 24.04 编译的，要求 glibc ≥ **2.38**
- 容器内 glibc 2.35 < 2.38，插件加载失败，NCCL 回退到 TCP Socket

**解决方案：自建镜像，内置 Ubuntu 22.04 专用的 ofi-nccl**

EFA installer 包含针对各 OS 版本编译的 deb 包。从 `aws-efa-installer-latest.tar.gz` 提取 Ubuntu 22.04 版本：

```bash
# 下载 EFA installer
curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz -o /tmp/efa-installer.tar.gz

# 提取 Ubuntu 22.04 的 deb 包
cd /tmp && tar -xzf efa-installer.tar.gz \
  aws-efa-installer/DEBS/UBUNTU2204/x86_64/libfabric1-aws_2.4.0amzn3.0_amd64.deb \
  aws-efa-installer/DEBS/UBUNTU2204/x86_64/libfabric-aws-dev_2.4.0amzn3.0_amd64.deb \
  aws-efa-installer/DEBS/UBUNTU2204/x86_64/libnccl-ofi_1.19.0-1_amd64.deb
```

**Dockerfile：**

```dockerfile
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2

# 复制 Ubuntu 22.04 专用的 EFA deb 包（libfabric 2.4.0 + ofi-nccl 1.19.0）
COPY libfabric1-aws_2.4.0amzn3.0_amd64.deb /tmp/
COPY libfabric-aws-dev_2.4.0amzn3.0_amd64.deb /tmp/
COPY libnccl-ofi_1.19.0-1_amd64.deb /tmp/

# 安装（覆盖原有的 ofi-nccl）
RUN dpkg -i /tmp/libfabric1-aws_2.4.0amzn3.0_amd64.deb \
            /tmp/libfabric-aws-dev_2.4.0amzn3.0_amd64.deb \
            /tmp/libnccl-ofi_1.19.0-1_amd64.deb && \
    rm /tmp/*.deb && ldconfig
```

新镜像中 `libnccl-net.so` 最高只依赖 GLIBC_2.29，完全兼容容器内 glibc 2.35。

**Build 并推送到 ECR：**

```bash
docker build -t pytorch-training-efa:2.10.0-cu130-ubuntu22.04 ./docker-build/

ECR_URI="940482411881.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-efa:2.10.0-cu130-ubuntu22.04"
docker tag pytorch-training-efa:2.10.0-cu130-ubuntu22.04 $ECR_URI
aws ecr get-login-password --region us-west-2 --profile myb300 | \
  docker login --username AWS --password-stdin 940482411881.dkr.ecr.us-west-2.amazonaws.com
docker push $ECR_URI
```

### 问题3：挂载宿主机 efa 目录覆盖了镜像内的 libfabric

**现象：** 使用新镜像但仍报 `GLIBC_2.38 not found`

**原因：** docker run 挂载了 `-v /opt/amazon/efa:/opt/amazon/efa`，把宿主机的 libfabric（需要 glibc 2.38）覆盖了镜像内安装的兼容版本。

**修复：** 使用新镜像时**不挂载** `/opt/amazon/efa`，让容器使用自己内置的 libfabric。

### 问题4：NCCL 初始化卡住（GDRCopy）

**现象：** NCCL channel 建立后卡住，日志停在 `Channel XX/0 : 0[0] -> 1[1] [send] via NET/Libfabric/0/GDRDMA`

**原因：** NCCL 尝试初始化 GDRCopy（GPU Direct Copy），但容器内没有 gdr 驱动，导致卡住。

**修复：** 加 `-e NCCL_P2P_DISABLE=1`

---

## 五、EFA 通信测试

### 测试脚本

```python
# /tmp/efa_simple_test.py
import os, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

t = torch.ones(256*1024, dtype=torch.float32, device=f"cuda:{local_rank}")  # 1MB
dist.all_reduce(t)
torch.cuda.synchronize()

if rank == 0:
    print("EFA all_reduce 1MB PASSED")
dist.destroy_process_group()
```

### 启动测试（使用新镜像）

```bash
NEW_IMAGE="940482411881.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-efa:2.10.0-cu130-ubuntu22.04"
IB_DEVICES=$(ls /dev/infiniband/ | grep -v "^by-" | sed 's|^|--device /dev/infiniband/|' | tr '\n' ' ')

# 节点2先启动（rank 1）
docker run --rm --runtime=nvidia --gpus all --network=host \
  --shm-size=64g --ulimit memlock=-1 \
  $IB_DEVICES \
  -v /tmp/efa_simple_test.py:/test.py \
  -e FI_EFA_USE_DEVICE_RDMA=1 -e FI_EFA_FORK_SAFE=1 \
  -e NCCL_SOCKET_IFNAME=enp71s0 -e NCCL_IB_HCA=rdmap \
  -e NCCL_DEBUG=WARN -e NCCL_P2P_DISABLE=1 \
  $NEW_IMAGE \
  torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
    --master_addr=172.31.47.177 --master_port=29509 /test.py

# 节点1随后启动（rank 0）
docker run --rm --runtime=nvidia --gpus all --network=host \
  --shm-size=64g --ulimit memlock=-1 \
  $IB_DEVICES \
  -v /tmp/efa_simple_test.py:/test.py \
  -e FI_EFA_USE_DEVICE_RDMA=1 -e FI_EFA_FORK_SAFE=1 \
  -e NCCL_SOCKET_IFNAME=enp71s0 -e NCCL_IB_HCA=rdmap \
  -e NCCL_DEBUG=WARN -e NCCL_P2P_DISABLE=1 \
  $NEW_IMAGE \
  torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
    --master_addr=172.31.47.177 --master_port=29509 /test.py
```

### 测试结果（2026-04-20）

```
NET/Plugin: Loaded net plugin Libfabric (v11)
NET/OFI Initializing aws-ofi-nccl 1.19.0
NET/OFI Using Libfabric version 2.4
NET/OFI Plugin selected platform: AWS
NET/OFI Using transport protocol RDMA (platform set)
NET/OFI Selected provider is efa, fabric is efa-direct (found 16 nics)
...
EFA all_reduce 1MB PASSED
```

---

## 六、SDXL 双机训练

### 训练镜像（使用新镜像）

```
940482411881.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-efa:2.10.0-cu130-ubuntu22.04
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

**节点2** `/opt/dlami/nvme/benchmark/config/accelerate_node2.yaml`：同上，`machine_rank: 1`

### 训练启动命令

```bash
NEW_IMAGE="940482411881.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-efa:2.10.0-cu130-ubuntu22.04"
IB_DEVICES=$(ls /dev/infiniband/ | grep -v "^by-" | sed 's|^|--device /dev/infiniband/|' | tr '\n' ' ')

# 节点1（先启动节点2，再启动节点1）
nohup docker run --rm \
  --runtime=nvidia --gpus all \
  --network=host \
  --shm-size=512g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  $IB_DEVICES \
  -v /opt/dlami/nvme/sdxl:/workspace \
  -v /opt/dlami/nvme/cache:/root/.cache \
  -v /opt/dlami/nvme/benchmark/config:/config \
  -v /opt/dlami/nvme/benchmark/train_node1_inner.sh:/train.sh \
  -e FI_EFA_USE_DEVICE_RDMA=1 \
  -e FI_EFA_FORK_SAFE=1 \
  -e NCCL_SOCKET_IFNAME=enp71s0 \
  -e NCCL_IB_HCA=rdmap \
  -e NCCL_DEBUG=WARN \
  -e NCCL_P2P_DISABLE=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -e OMP_NUM_THREADS=8 \
  $NEW_IMAGE \
  bash /train.sh \
  > /opt/dlami/nvme/sdxl/train_node1.log 2>&1 &
```

**关键变化（与旧命令的区别）：**
- 使用新镜像 `pytorch-training-efa`（内置兼容的 ofi-nccl）
- **不挂载** `/opt/amazon/ofi-nccl` 和 `/opt/amazon/efa`
- **不设置** `LD_LIBRARY_PATH` 中的 ofi-nccl/efa 路径
- 新增 `-e NCCL_P2P_DISABLE=1`

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

---

## 七、故障排查汇总

| 错误 | 原因 | 修复 |
|------|------|------|
| `IBV_WC_REM_ACCESS_ERR` | NCCL 选了 NVSwitch IB 网卡 | `-e NCCL_IB_HCA=rdmap` |
| `GLIBC_2.38 not found` | 容器 glibc 2.35 < 宿主机 ofi-nccl 要求的 2.38 | 使用新镜像（内置 Ubuntu 22.04 版 ofi-nccl） |
| 新镜像仍报 `GLIBC_2.38` | 挂载了宿主机 `/opt/amazon/efa` 覆盖了镜像内 libfabric | 不挂载 `/opt/amazon/efa` |
| NCCL 初始化卡住 | GDRCopy 初始化失败导致阻塞 | `-e NCCL_P2P_DISABLE=1` |
| `fi_info` 命令找不到 | 路径问题 | 使用 `/opt/amazon/efa/bin/fi_info` |
| rendezvous timeout | 节点2启动太慢 | 先启动节点2，再启动节点1 |
| pip install 不工作 | 镜像内 pip 有 bug | 使用 `python -m pip install` |

---

## 八、双机训练架构图

```
p6-b300 双机 16卡 DDP 训练架构
═══════════════════════════════════════════════════════════════════════════════════════════

  HuggingFace (SDXL Base 1.0 + Dataset)
        │ model weights + dataset
        ▼
┌─────────────────────────────────────────┐        ┌─────────────────────────────────────────┐
│  Node 1  172.31.47.177  (master)        │        │  Node 2  172.31.35.107                  │
│  machine_rank=0                         │        │  machine_rank=1                         │
│                                         │        │                                         │
│  ┌─────────────────────────────────┐    │        │  ┌─────────────────────────────────┐    │
│  │  8x NVIDIA B300 GPU (NVLink)    │    │        │  │  8x NVIDIA B300 GPU (NVLink)    │    │
│  │                                 │    │        │  │                                 │    │
│  │  [rank0] [rank1] [rank2] [rank3]│    │        │  │  [rank8] [rank9][rank10][rank11]│    │
│  │  [rank4] [rank5] [rank6] [rank7]│    │        │  │ [rank12][rank13][rank14][rank15]│    │
│  │  ◄────── NVLink all_reduce ────►│    │        │  │  ◄────── NVLink all_reduce ────►│    │
│  └──────────────┬──────────────────┘    │        │  └──────────────┬──────────────────┘    │
│                 │                       │        │                 │                       │
│  ┌──────────────▼──────────────────┐    │        │  ┌──────────────▼──────────────────┐    │
│  │  NCCL 2.28.9                    │    │        │  │  NCCL 2.28.9                    │    │
│  │  ofi-nccl 1.19.0                │    │        │  │  ofi-nccl 1.19.0                │    │
│  │  libfabric 2.4.0                │    │        │  │  libfabric 2.4.0                │    │
│  └──────────────┬──────────────────┘    │        │  └──────────────┬──────────────────┘    │
│                 │ EFA provider          │        │                 │ EFA provider          │
│  ┌──────────────▼──────────────────┐    │        │  ┌──────────────▼──────────────────┐    │
│  │  16x EFA NICs                   │◄───┼────────┼──┤  16x EFA NICs                   │    │
│  │  rdmap86s0 ~ rdmap192s0         │    │        │  │  rdmap86s0 ~ rdmap192s0         │    │
│  │  ~27 Gbps/NIC  Total ~438 Gbps  │────┼────────┼──►  ~27 Gbps/NIC  Total ~438 Gbps  │    │
│  └──────────────┬──────────────────┘    │        │  └──────────────┬──────────────────┘    │
│                 │                       │        │                 │                       │
│  ┌──────────────▼──────────────────┐    │        │  ┌──────────────▼──────────────────┐    │
│  │  ENA enp71s0                    │◄ ─ ┼ ─ ─ ─ ┼─ ┤  ENA enp71s0                    │    │
│  │  rendezvous port 29500  (TCP)   │ ─ ─┼ ─ ─ ─ ┼──►  rendezvous port 29500  (TCP)   │    │
│  └─────────────────────────────────┘    │        │  └─────────────────────────────────┘    │
│                                         │        │                                         │
│  NVMe /opt/dlami/nvme/sdxl/checkpoints  │        │  (no checkpoint, rank != 0)             │
│  checkpoint-100/200/300/400/500         │        │                                         │
└─────────────────────────────────────────┘        └─────────────────────────────────────────┘

                    ◄──────── EFA RDMA Write all_reduce ────────►
                         ~438 Gbps 双向  |  retrans=0  |  drops=0

═══════════════════════════════════════════════════════════════════════════════════════════
训练参数: SDXL instruct-pix2pix  |  batch=128 (16GPU×2×4)  |  500 steps  |  bf16  |  ~3s/it
```

---

## 九、自定义 Docker 镜像（Dockerfile）

完整 Dockerfile 见仓库根目录 `Dockerfile`，内容如下：

```dockerfile
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2

# 复制 Ubuntu 22.04 专用的 EFA deb 包（libfabric 2.4.0 + ofi-nccl 1.19.0）
# 从 aws-efa-installer-latest.tar.gz 的 DEBS/UBUNTU2204/x86_64/ 目录提取
COPY libfabric1-aws_2.4.0amzn3.0_amd64.deb /tmp/
COPY libfabric-aws-dev_2.4.0amzn3.0_amd64.deb /tmp/
COPY libnccl-ofi_1.19.0-1_amd64.deb /tmp/

# 安装（覆盖原有的 ofi-nccl，兼容容器内 glibc 2.35）
RUN dpkg -i /tmp/libfabric1-aws_2.4.0amzn3.0_amd64.deb \
            /tmp/libfabric-aws-dev_2.4.0amzn3.0_amd64.deb \
            /tmp/libnccl-ofi_1.19.0-1_amd64.deb && \
    rm /tmp/*.deb && ldconfig
```

### Build 步骤

```bash
# 1. 下载 EFA installer 并提取 Ubuntu 22.04 deb 包
curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz -o /tmp/efa-installer.tar.gz
cd /tmp && tar -xzf efa-installer.tar.gz \
  aws-efa-installer/DEBS/UBUNTU2204/x86_64/libfabric1-aws_2.4.0amzn3.0_amd64.deb \
  aws-efa-installer/DEBS/UBUNTU2204/x86_64/libfabric-aws-dev_2.4.0amzn3.0_amd64.deb \
  aws-efa-installer/DEBS/UBUNTU2204/x86_64/libnccl-ofi_1.19.0-1_amd64.deb
cp aws-efa-installer/DEBS/UBUNTU2204/x86_64/*.deb ./docker-build/

# 2. Build 镜像
aws ecr get-login-password --region us-west-2 --profile myb300 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com
docker build -t pytorch-training-efa:2.10.0-cu130-ubuntu22.04 ./docker-build/

# 3. Push 到私有 ECR
ECR_URI="940482411881.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-efa:2.10.0-cu130-ubuntu22.04"
aws ecr get-login-password --region us-west-2 --profile myb300 | \
  docker login --username AWS --password-stdin 940482411881.dkr.ecr.us-west-2.amazonaws.com
docker tag pytorch-training-efa:2.10.0-cu130-ubuntu22.04 $ECR_URI
docker push $ECR_URI
```

### 镜像地址

```
940482411881.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-efa:2.10.0-cu130-ubuntu22.04
```

---

## 十、glibc 版本分析与最终解决方案

### 版本对比

| 环境 | OS | glibc 版本 |
|------|-----|-----------|
| EC2 宿主机 | Ubuntu 24.04 | **2.39** |
| 容器镜像（pytorch-training:2.10.0...ubuntu22.04） | Ubuntu 22.04 | **2.35** |
| 宿主机 ofi-nccl 1.18.0（`/opt/amazon/ofi-nccl`） | 为 Ubuntu 24.04 编译 | 要求 **≥ 2.38** |
| 自建镜像内置 ofi-nccl 1.19.0（Ubuntu 22.04 专用） | 为 Ubuntu 22.04 编译 | 要求 **≥ 2.16** ✅ |

### 问题链路

```
宿主机 ofi-nccl 1.18.0 (需要 glibc 2.38)
    挂载进容器 (-v /opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl)
        容器内 glibc 只有 2.35
            → libnccl-net.so 加载失败
                → NCCL 回退到 TCP Socket (NET/Socket)
                    → 跨节点通信走 ENA 网卡，带宽极低
```

### 解决方案

不挂载宿主机的 ofi-nccl，改用自建镜像内置的 Ubuntu 22.04 专用版本（ofi-nccl 1.19.0，最高依赖 glibc 2.16）。

**关键：使用新镜像时不能挂载 `/opt/amazon/efa`**，否则宿主机的 libfabric.so（需要 glibc 2.38）会覆盖镜像内兼容的版本，导致同样的问题。

### 验证结果

```
# NCCL_DEBUG=INFO 输出确认走 EFA：
NET/Plugin: Loaded net plugin Libfabric (v11)
NET/OFI Initializing aws-ofi-nccl 1.19.0
NET/OFI Using Libfabric version 2.4
NET/OFI Plugin selected platform: AWS
NET/OFI Using transport protocol RDMA (platform set)
NET/OFI Selected provider is efa, fabric is efa-direct (found 16 nics)

# EFA 带宽监控确认（训练中）：
rdmap101s0   27.39 Gbps TX   27.39 Gbps RX
rdmap102s0   27.39 Gbps TX   27.39 Gbps RX
... (16卡全部活跃，总计 ~438 Gbps 双向)
```

---

## 十一、最终训练启动命令

### 节点2 先启动（通过 SSM）

```bash
NEW_IMAGE="940482411881.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-efa:2.10.0-cu130-ubuntu22.04"
IB_DEVICES=$(ls /dev/infiniband/ | grep -v "^by-" | sed 's|^|--device /dev/infiniband/|' | tr '\n' ' ')

nohup docker run --rm \
  --runtime=nvidia --gpus all \
  --network=host \
  --shm-size=512g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  $IB_DEVICES \
  -v /opt/dlami/nvme/sdxl:/workspace \
  -v /opt/dlami/nvme/cache:/root/.cache \
  -v /opt/dlami/nvme/benchmark/config:/config \
  -v /opt/dlami/nvme/benchmark/train_node2_inner.sh:/train.sh \
  -e FI_EFA_USE_DEVICE_RDMA=1 \
  -e FI_EFA_FORK_SAFE=1 \
  -e NCCL_SOCKET_IFNAME=enp71s0 \
  -e NCCL_IB_HCA=rdmap \
  -e NCCL_DEBUG=WARN \
  -e NCCL_P2P_DISABLE=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -e OMP_NUM_THREADS=8 \
  $NEW_IMAGE \
  bash /train.sh \
  > /opt/dlami/nvme/sdxl/train_node2.log 2>&1 &
```

### 节点1 随后启动（主节点，本机执行）

```bash
NEW_IMAGE="940482411881.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-efa:2.10.0-cu130-ubuntu22.04"
IB_DEVICES=$(ls /dev/infiniband/ | grep -v "^by-" | sed 's|^|--device /dev/infiniband/|' | tr '\n' ' ')

nohup docker run --rm \
  --runtime=nvidia --gpus all \
  --network=host \
  --shm-size=512g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  $IB_DEVICES \
  -v /opt/dlami/nvme/sdxl:/workspace \
  -v /opt/dlami/nvme/cache:/root/.cache \
  -v /opt/dlami/nvme/benchmark/config:/config \
  -v /opt/dlami/nvme/benchmark/train_node1_inner.sh:/train.sh \
  -e FI_EFA_USE_DEVICE_RDMA=1 \
  -e FI_EFA_FORK_SAFE=1 \
  -e NCCL_SOCKET_IFNAME=enp71s0 \
  -e NCCL_IB_HCA=rdmap \
  -e NCCL_DEBUG=WARN \
  -e NCCL_P2P_DISABLE=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -e OMP_NUM_THREADS=8 \
  $NEW_IMAGE \
  bash /train.sh \
  > /opt/dlami/nvme/sdxl/train_node1.log 2>&1 &
```

### 两个命令的唯一区别

| | 节点1 | 节点2 |
|--|--|--|
| 挂载脚本 | `train_node1_inner.sh` | `train_node2_inner.sh` |
| accelerate config | machine_rank=0，主节点 | machine_rank=1 |
| 其他所有参数 | 完全相同 | 完全相同 |

### 注意事项

1. **不挂载** `/opt/amazon/efa` 和 `/opt/amazon/ofi-nccl`（用镜像内置版本）
2. **不设置** `LD_LIBRARY_PATH` 中的 efa/ofi-nccl 路径
3. 先启动节点2，再启动节点1（节点1作为 rendezvous server 等待节点2连接）
4. 节点2需在节点1启动后 15 分钟内完成 pip 安装并连接
