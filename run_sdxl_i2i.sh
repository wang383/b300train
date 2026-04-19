#!/bin/bash
set -e

IMAGE="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-ec2"
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
