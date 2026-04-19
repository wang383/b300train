# SDXL InstructPix2Pix 训练完整指南

## 环境信息

| 项目 | 值 |
|------|-----|
| EC2 实例类型 | p6-b300.48xlarge |
| 支持实例 | G4dn, G5, G6, Gr6, G6e, P4d, P4de, P5, P5e, P5en, P6-B200, P6-B300 |
| 操作系统 | Ubuntu 22.04.5 LTS |
| 架构 | x86_64 |
| 内核版本 | 6.8.0-1050-aws |
| Python | /usr/bin/python3.10 |
| NVIDIA 驱动 | 580.126.09（nvidia-smi 显示 595.58.03） |
| CUDA | /usr/local/cuda-13.0（CUDA 13.2） |
| nvidia-container-toolkit | 1.19.0 |
| DCGM | 4.5.2 |
| EFA 版本 | 1.45.0 |
| OFI-NCCL 版本 | 1.17.2 |
| NVMe 挂载点 | /opt/dlami/nvme（28TB LVM，8x NVMe 组成） |
| EBS 类型 | gp3 |
| GPU | 8x NVIDIA B300 SXM6 AC（每张 275GB HBM3e，148 SM） |

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
| Ubuntu | 22.04 |
| 镜像大小 | ~27GB |

---

## 训练任务说明

| 项目 | 值 |
|------|-----|
| 任务类型 | InstructPix2Pix（指令驱动图生图） |
| 基础模型 | Stable Diffusion XL Base 1.0 |
| 模型来源 | HuggingFace: `stabilityai/stable-diffusion-xl-base-1.0` |
| 模型大小 | ~13 GB |
| 训练脚本 | diffusers 官方 `train_instruct_pix2pix_sdxl.py` |
| 数据集 | `fusing/instructpix2pix-1000-samples`（HuggingFace，398MB） |
| 训练精度 | bf16 |
| GPU 数量 | 8（DDP） |
| 训练步数 | 500 steps |

---

## 目录结构

```
/opt/dlami/nvme/
├── cache/
│   └── huggingface/
│       └── hub/
│           ├── models--stabilityai--stable-diffusion-xl-base-1.0/  # 13GB 模型
│           └── datasets--fusing--instructpix2pix-1000-samples/     # 398MB 数据集
├── sdxl/
│   ├── checkpoints/          # 训练 checkpoint（每100步保存）
│   ├── logs/                 # TensorBoard 日志
│   ├── train.log             # 训练输出日志
│   └── train_instruct_pix2pix_sdxl.py  # 训练脚本
└── benchmark/
    └── run_sdxl_i2i.sh       # 启动脚本
```

---

## 完整复现步骤

### 第一步：准备目录

```bash
mkdir -p /opt/dlami/nvme/sdxl/{checkpoints,logs}
mkdir -p /opt/dlami/nvme/cache
```

### 第二步：登录 ECR 并拉取镜像

```bash
# 获取当前 region
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')" \
  http://169.254.169.254/latest/meta-data/placement/region)

# 登录 ECR
aws ecr get-login-password --region ${REGION} | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com

# 拉取镜像（约 27GB，需要一些时间）
docker pull 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2
```

### 第三步：确认 EFA 设备

```bash
# 确认 EFA 接口存在
ls /dev/infiniband/
# 应该看到: uverbs0 uverbs1 rdma_cm umad0 umad1

# 确认 EFA 库存在
ls /opt/amazon/efa/lib/
ls /opt/amazon/ofi-nccl/lib/
```

### 第四步：创建训练启动脚本

```bash
cat > /opt/dlami/nvme/benchmark/run_sdxl_i2i.sh << 'SCRIPT'
#!/bin/bash
set -e

REGION=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')" \
  http://169.254.169.254/latest/meta-data/placement/region)

IMAGE="763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2"
WORK_DIR="/opt/dlami/nvme/sdxl"
mkdir -p ${WORK_DIR}/{checkpoints,logs}

docker run --rm \
  --runtime=nvidia \
  --gpus all \
  --shm-size=512g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network=host \
  --device /dev/infiniband/uverbs0 \
  --device /dev/infiniband/uverbs1 \
  --device /dev/infiniband/rdma_cm \
  --device /dev/infiniband/umad0 \
  --device /dev/infiniband/umad1 \
  -v /opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl \
  -v /opt/amazon/efa:/opt/amazon/efa \
  -v ${WORK_DIR}:/workspace \
  -v /opt/dlami/nvme/cache:/root/.cache \
  -e LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:/usr/local/lib \
  -e FI_EFA_USE_DEVICE_RDMA=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -e NCCL_DEBUG=WARN \
  -e OMP_NUM_THREADS=8 \
  ${IMAGE} \
  bash -c "
    pip install accelerate transformers datasets bitsandbytes -q &&
    pip install git+https://github.com/huggingface/diffusers.git -q &&

    # 下载 diffusers 官方训练脚本
    if [ ! -f /workspace/train_instruct_pix2pix_sdxl.py ]; then
      wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/instruct_pix2pix/train_instruct_pix2pix_sdxl.py \
        -O /workspace/train_instruct_pix2pix_sdxl.py
    fi

    # 配置 accelerate（8 GPU DDP）
    cat > /tmp/accelerate_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

    accelerate launch \
      --config_file /tmp/accelerate_config.yaml \
      /workspace/train_instruct_pix2pix_sdxl.py \
      --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
      --dataset_name fusing/instructpix2pix-1000-samples \
      --use_ema \
      --resolution 1024 \
      --train_batch_size 2 \
      --gradient_accumulation_steps 4 \
      --gradient_checkpointing \
      --max_train_steps 500 \
      --learning_rate 5e-6 \
      --lr_scheduler cosine \
      --lr_warmup_steps 50 \
      --mixed_precision bf16 \
      --output_dir /workspace/checkpoints \
      --report_to tensorboard \
      --logging_dir /workspace/logs \
      --checkpointing_steps 100 \
      --seed 42 \
      2>&1 | tee /workspace/train.log
  "
SCRIPT

chmod +x /opt/dlami/nvme/benchmark/run_sdxl_i2i.sh
```

### 第五步：执行训练

```bash
bash /opt/dlami/nvme/benchmark/run_sdxl_i2i.sh
```

---

## 数据和模型下载说明

模型和数据集**无需手动下载**，训练脚本启动后自动从 HuggingFace 下载到 NVMe：

- 模型下载路径：`/opt/dlami/nvme/cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/`
- 数据集下载路径：`/opt/dlami/nvme/cache/huggingface/hub/datasets--fusing--instructpix2pix-1000-samples/`

通过 `-v /opt/dlami/nvme/cache:/root/.cache` 和 `-e HF_HOME=/root/.cache/huggingface` 将缓存目录挂载到 NVMe，确保：
1. 下载的文件写入 NVMe 而非系统盘
2. 重复运行时直接复用缓存，无需重新下载

---

## EFA 加载说明

EFA（Elastic Fabric Adapter）通过以下方式注入容器：

```bash
# 挂载 EFA 设备
--device /dev/infiniband/uverbs0 \
--device /dev/infiniband/uverbs1 \
--device /dev/infiniband/rdma_cm \
--device /dev/infiniband/umad0 \
--device /dev/infiniband/umad1 \

# 挂载 EFA 和 OFI-NCCL 库
-v /opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl \
-v /opt/amazon/efa:/opt/amazon/efa \

# 设置库路径和 EFA 环境变量
-e LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:/usr/local/lib \
-e FI_EFA_USE_DEVICE_RDMA=1 \
```

`FI_EFA_USE_DEVICE_RDMA=1` 启用 EFA 的 device RDMA 模式，允许 GPU 直接通过 EFA 传输数据，绕过 CPU，降低 NCCL all-reduce 延迟。

---

## 训练过程关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--train_batch_size` | 2 | 每 GPU batch size |
| `--gradient_accumulation_steps` | 4 | 梯度累积，等效全局 batch = 2×8×4=64 |
| `--mixed_precision` | bf16 | B300 推荐 bf16，性能优于 fp16 |
| `--gradient_checkpointing` | 开启 | 节省显存，以计算换内存 |
| `--resolution` | 1024 | SDXL 原生分辨率 |
| `--use_ema` | 开启 | 指数移动平均，提升模型稳定性 |
| `--checkpointing_steps` | 100 | 每 100 步保存一次 checkpoint |
| `--shm-size` | 512g | 多 GPU NCCL 通信需要大共享内存 |
| `--ulimit memlock=-1` | 无限制 | EFA/NCCL 需要锁定内存 |

---

## 训练性能观测

```bash
# 查看训练进度
tail -f /opt/dlami/nvme/sdxl/train.log

# 查看 GPU 利用率
nvidia-smi dmon -s u -c 5

# 查看 SM 活跃率（需要 dcgmi）
nvidia-smi dmon -e 1001,1002,1003,1004 -c 3
# GRACT: GPU活跃率  SMACT: SM活跃率  SMOCC: SM占用率  TENSO: TensorCore利用率

# 查看 checkpoint
ls -lh /opt/dlami/nvme/sdxl/checkpoints/
```

---

## 实测性能数据

| 指标 | 值 |
|------|-----|
| 训练速度 | ~4 秒/step |
| GPU 活跃率（GRACT） | 96~100% |
| SM 活跃率（SMACT） | ~56% |
| SM 占用率（SMOCC） | ~25% |
| Tensor Core 利用率 | ~12% |
| 显存使用 | ~161 GB / 275 GB（每张） |
| 功耗 | 395~414W / 1100W |
| 500 steps 总耗时 | ~35 分钟 |

SM 利用率偏低的原因：batch size 较小（2/GPU），可调大至 8 提升利用率。

---

## 注意事项

1. **NVMe 是临时存储**：实例停止后数据丢失，训练完成后及时将 checkpoint 同步到 S3：
   ```bash
   aws s3 sync /opt/dlami/nvme/sdxl/checkpoints/ s3://your-bucket/sdxl-checkpoints/
   ```

2. **首次运行需要下载**：模型 13GB + 数据集 398MB，需要网络时间，后续运行直接使用缓存。

3. **ECR 镜像 region 需匹配**：`763104351884.dkr.ecr.<region>.amazonaws.com`，region 需与 EC2 所在 region 一致。

4. **EFA 设备路径**：不同实例 EFA 接口数量可能不同，运行前确认 `ls /dev/infiniband/`。
