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
