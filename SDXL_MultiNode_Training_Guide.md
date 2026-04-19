# SDXL 多机多卡训练指南（2台 p6-b300.48xlarge，16卡）

## 重要说明：EFA 网卡配置

p6-b300.48xlarge 支持 6400 Gbps EFA 网络，但 **EFA 网卡必须在启动实例时配置**，不能事后添加。

### 启动实例时配置 EFA（CLI 方式）

```bash
aws ec2 run-instances \
  --instance-type p6-b300.48xlarge \
  --image-id ami-06764d4b3d62da4fa \
  --key-name myp300 \
  --count 2 \
  --network-interfaces \
    "NetworkCardIndex=0,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=interface" \
    "NetworkCardIndex=1,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=2,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=3,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=4,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=5,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=6,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=7,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=8,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=9,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=10,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=11,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=12,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=13,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=14,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=15,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
    "NetworkCardIndex=16,DeviceIndex=0,Groups=<sg-id>,SubnetId=<subnet-id>,InterfaceType=efa-only" \
  --capacity-reservation-specification CapacityReservationTarget={CapacityReservationId=<cr-id>} \
  --iam-instance-profile Name=b300test \
  --region us-west-2
```

### 控制台配置 EFA

1. EC2 → Launch Instance → 选择 AMI 和 `p6-b300.48xlarge`
2. **Network settings** → **Edit** → **Advanced network configuration**
3. 网卡 0 保持 `interface`（ENA，用于 SSH）
4. 点 **Add network interface**，依次添加 NetworkCardIndex 1-16，Interface type 选 **efa-only**
5. 验证 EFA 是否配置成功：`fi_info -p efa`（应看到 efa provider）

---

## 环境信息

| 项目 | 值 |
|------|-----|
| 实例类型 | p6-b300.48xlarge |
| AMI | Deep Learning Base AMI with Single CUDA (Ubuntu 24.04) |
| AMI ID | ami-06764d4b3d62da4fa |
| GPU | 8x NVIDIA B300 SXM6 AC（每张 275GB HBM3e） |
| CUDA | 13.0 |
| NVIDIA 驱动 | 595.58.03 |
| EFA 版本 | 1.45.0（需在启动时配置 EFA 网卡） |
| OFI-NCCL | 1.17.2 |
| NVMe | /opt/dlami/nvme（28TB，关机后数据丢失） |

---

## 训练容器镜像

```
763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2
```

| 组件 | 版本 |
|------|------|
| PyTorch | 2.10.0 |
| CUDA | 13.0 |
| Python | 3.13 |
| 镜像大小 | ~27GB |

---

## 前提条件

- 两台实例在同一 VPC、子网、AZ
- 安全组允许两台机器互相访问所有流量
- 两台机器都已配置 EFA 网卡（启动时）
- IAM 角色有 SSM 权限（AmazonSSMManagedInstanceCore）

---

## 完整复现步骤

### 第一步：准备目录（两台机器都执行）

```bash
mkdir -p /opt/dlami/nvme/sdxl/{checkpoints,logs}
mkdir -p /opt/dlami/nvme/cache
mkdir -p /opt/dlami/nvme/benchmark/config
```

### 第二步：登录 ECR 并拉取镜像（两台机器都执行，约 27GB）

```bash
REGION=us-west-2
aws ecr get-login-password --region ${REGION} | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com

docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2
```

### 第三步：验证 EFA（两台机器都执行）

```bash
# 确认 EFA 设备存在
ls /dev/infiniband/
# 应看到多个 uverbs 设备

# 确认 EFA provider 可用
/opt/amazon/efa/bin/fi_info -p efa | head -5
# 应看到 provider: efa
```

### 第四步：配置安全组（节点间互通）

```bash
# 允许节点1和节点2互相访问所有流量
aws ec2 authorize-security-group-ingress \
  --group-id <节点1-sg-id> \
  --ip-permissions IpProtocol=-1,IpRanges=[{CidrIp=<节点2-内网IP>/32}] \
  --region us-west-2

aws ec2 authorize-security-group-ingress \
  --group-id <节点2-sg-id> \
  --ip-permissions IpProtocol=-1,IpRanges=[{CidrIp=<节点1-内网IP>/32}] \
  --region us-west-2
```

### 第五步：生成 accelerate 配置文件

**节点1（主节点，MASTER_IP 替换为节点1内网 IP）：**

```bash
MASTER_IP=<节点1内网IP>

cat > /opt/dlami/nvme/benchmark/config/accelerate_node1.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
machine_rank: 0
num_machines: 2
num_processes: 16
main_process_ip: ${MASTER_IP}
main_process_port: 29500
mixed_precision: bf16
same_network: true
use_cpu: false
EOF
```

**节点2：**

```bash
MASTER_IP=<节点1内网IP>

cat > /opt/dlami/nvme/benchmark/config/accelerate_node2.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
machine_rank: 1
num_machines: 2
num_processes: 16
main_process_ip: ${MASTER_IP}
main_process_port: 29500
mixed_precision: bf16
same_network: true
use_cpu: false
EOF
```

### 第六步：生成训练内部脚本

**节点1（/opt/dlami/nvme/benchmark/train_node1_inner.sh）：**

```bash
cat > /opt/dlami/nvme/benchmark/train_node1_inner.sh << 'EOF'
#!/bin/bash
set -e
python -m pip install accelerate transformers datasets bitsandbytes
python -m pip install git+https://github.com/huggingface/diffusers.git

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
EOF
chmod +x /opt/dlami/nvme/benchmark/train_node1_inner.sh
```

**节点2（/opt/dlami/nvme/benchmark/train_node2_inner.sh）：**

```bash
cat > /opt/dlami/nvme/benchmark/train_node2_inner.sh << 'EOF'
#!/bin/bash
set -e
python -m pip install accelerate transformers datasets bitsandbytes
python -m pip install git+https://github.com/huggingface/diffusers.git

wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/instruct_pix2pix/train_instruct_pix2pix_sdxl.py \
  -O /workspace/train_instruct_pix2pix_sdxl.py

accelerate launch --config_file /config/accelerate_node2.yaml \
  /workspace/train_instruct_pix2pix_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --dataset_name fusing/instructpix2pix-1000-samples \
  --use_ema --resolution 1024 --train_batch_size 2 \
  --gradient_accumulation_steps 4 --gradient_checkpointing \
  --max_train_steps 500 --learning_rate 5e-6 \
  --lr_scheduler cosine --lr_warmup_steps 50 --mixed_precision bf16 \
  --output_dir /workspace/checkpoints --checkpointing_steps 100 --seed 42 \
  2>&1 | tee /workspace/train_node2.log
EOF
chmod +x /opt/dlami/nvme/benchmark/train_node2_inner.sh
```

### 第七步：启动训练

**先启动节点1（主节点）：**

```bash
nohup docker run --rm \
  --runtime=nvidia --gpus all \
  --shm-size=512g --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  --device /dev/infiniband/uverbs0 \
  --device /dev/infiniband/uverbs1 \
  --device /dev/infiniband/rdma_cm \
  --device /dev/infiniband/umad0 \
  --device /dev/infiniband/umad1 \
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
  -e NCCL_DEBUG=WARN \
  -e HF_HOME=/root/.cache/huggingface \
  -e OMP_NUM_THREADS=8 \
  763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2 \
  bash /train.sh > /opt/dlami/nvme/sdxl/train_node1.log 2>&1 &
echo "节点1 PID: $!"
```

**节点1启动后，立即启动节点2：**

```bash
nohup docker run --rm \
  --runtime=nvidia --gpus all \
  --shm-size=512g --ulimit memlock=-1 --ulimit stack=67108864 \
  --network=host \
  --device /dev/infiniband/uverbs0 \
  --device /dev/infiniband/uverbs1 \
  --device /dev/infiniband/rdma_cm \
  --device /dev/infiniband/umad0 \
  --device /dev/infiniband/umad1 \
  -v /opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl \
  -v /opt/amazon/efa:/opt/amazon/efa \
  -v /opt/dlami/nvme/sdxl:/workspace \
  -v /opt/dlami/nvme/cache:/root/.cache \
  -v /opt/dlami/nvme/benchmark/config:/config \
  -v /opt/dlami/nvme/benchmark/train_node2_inner.sh:/train.sh \
  -e LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:/usr/local/lib \
  -e FI_EFA_USE_DEVICE_RDMA=1 \
  -e FI_EFA_FORK_SAFE=1 \
  -e NCCL_SOCKET_IFNAME=enp71s0 \
  -e NCCL_DEBUG=WARN \
  -e HF_HOME=/root/.cache/huggingface \
  -e OMP_NUM_THREADS=8 \
  763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2 \
  bash /train.sh > /opt/dlami/nvme/sdxl/train_node2.log 2>&1 &
echo "节点2 PID: $!"
```

---

## 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_machines` | 2 | 两台机器 |
| `num_processes` | 16 | 每台 8 GPU，共 16 进程 |
| `machine_rank` | 0/1 | 节点1=0，节点2=1 |
| `main_process_ip` | 节点1内网IP | rendezvous 地址 |
| `main_process_port` | 29500 | rendezvous 端口 |
| `FI_EFA_USE_DEVICE_RDMA` | 1 | 启用 EFA RDMA |
| `FI_EFA_FORK_SAFE` | 1 | 多进程 fork 安全 |
| `NCCL_SOCKET_IFNAME` | enp71s0 | NCCL bootstrap 网卡名 |

---

## 监控训练

```bash
# 查看节点1训练日志
tail -f /opt/dlami/nvme/sdxl/train_node1.log

# 查看 GPU 利用率
nvidia-smi dmon -s u -d 1

# 查看 checkpoint
ls -lh /opt/dlami/nvme/sdxl/checkpoints/
```

---

## 常见问题

### 1. `fi_info -p efa` 返回空
原因：实例启动时没有配置 EFA 网卡。需要重新启动实例并在 `--network-interfaces` 中添加 `InterfaceType=efa-only` 的网卡（NetworkCardIndex 1-16）。

### 2. NCCL `IBV_WC_REM_ACCESS_ERR`
原因：IB RDMA 内存访问错误，通常是 GPU 与 HCA 的 PCIe 距离过远导致 GPU Direct RDMA 失败。
解决：设置 `NCCL_IB_GDR_LEVEL=0` 禁用 GPU Direct RDMA，或排除有问题的 GPU（`CUDA_VISIBLE_DEVICES`）。

### 3. `pip install datasets` 不工作
原因：镜像内 pip 26.0.1 有 bug，`pip install` 命令解析异常。
解决：使用 `python -m pip install` 替代 `pip install`。

### 4. rendezvous timeout
原因：节点2启动太慢，节点1等待超时（默认 900 秒）。
解决：确保节点2在节点1启动后 15 分钟内完成 pip 安装并连接。

### 5. SSM 执行 docker 命令时单引号被截断
原因：SSM commands 数组中单引号在 JSON 序列化时被截断。
解决：将 bash -c 的内容写成独立脚本文件，通过 `-v` 挂载进容器执行。

---

## 注意事项

1. **NVMe 是临时存储**：实例停止后数据丢失，训练完成后及时同步到 S3：
   ```bash
   aws s3 sync /opt/dlami/nvme/sdxl/checkpoints/ s3://your-bucket/sdxl-checkpoints/
   ```

2. **启动顺序**：先启动节点1，再启动节点2，节点2会等待节点1的 rendezvous

3. **Checkpoint 只保存在节点1**（rank 0），节点2不写 checkpoint

4. **网卡名可能不同**：`NCCL_SOCKET_IFNAME` 需要设置为实际的 ENA 网卡名，用 `ip link show` 查看
