# P6-B300 EFA/NCCL 实例部署文档

## 环境信息

| 项目 | 值 |
|------|-----|
| 区域 | us-west-2 |
| 可用区 | us-west-2a (usw2-az2) |
| 账号 | 940482411881 |
| Capacity Block CR | cr-0d34cf36b15d769e0 |
| 实例类型 | p6-b300.48xlarge |
| AMI | ami-06764d4b3d62da4fa |
| 密钥对 | myp300 |

---

## 步骤一：创建集群置放群组

```bash
aws ec2 create-placement-group \
  --group-name p6-efa-nccl-pg \
  --strategy cluster \
  --tag-specifications 'ResourceType=placement-group,Tags=[{Key=Name,Value=p6-efa-nccl-pg}]' \
  --region us-west-2
```

- 置放群组 ID：`pg-0b5d2db225e12a834`
- 策略：`cluster`（确保实例物理上靠近，最大化网络性能）

---

## 步骤二：创建 EFA 安全组

### 2.1 创建安全组

```bash
aws ec2 create-security-group \
  --group-name efa-nccl-sg \
  --description "EFA enabled security group for NCCL workloads" \
  --vpc-id vpc-04d237c717632ae38 \
  --region us-west-2
```

- 安全组 ID：`sg-0957308fc5a969ef0`
- VPC：`vpc-04d237c717632ae38`（默认 VPC）

### 2.2 配置入站规则

```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-0957308fc5a969ef0 \
  --ip-permissions \
    '{"IpProtocol":"-1","UserIdGroupPairs":[{"GroupId":"sg-0957308fc5a969ef0"}]}' \
    '{"FromPort":22,"IpProtocol":"tcp","IpRanges":[{"CidrIp":"0.0.0.0/0"}],"ToPort":22}' \
  --region us-west-2
```

| 规则 | 说明 |
|------|------|
| 所有流量，来源=自身安全组 | EFA 节点间通信（NCCL 要求） |
| TCP 22，来源=0.0.0.0/0 | SSH 登录 |

### 2.3 配置出站规则

```bash
# 允许组内所有流量（EFA/NCCL 节点间通信）
aws ec2 authorize-security-group-egress \
  --group-id sg-0957308fc5a969ef0 \
  --ip-permissions \
    '{"IpProtocol":"-1","UserIdGroupPairs":[{"GroupId":"sg-0957308fc5a969ef0"}]}' \
  --region us-west-2

# 允许 HTTPS 出站（SSM Agent、AWS API 等）
aws ec2 authorize-security-group-egress \
  --group-id sg-0957308fc5a969ef0 \
  --ip-permissions \
    '{"FromPort":443,"IpProtocol":"tcp","IpRanges":[{"CidrIp":"0.0.0.0/0"}],"ToPort":443}' \
  --region us-west-2
```

---

## 步骤三：启动实例

### P6-B300 网卡规格说明

p6-b300.48xlarge 有 **17 个网卡（NCI 0-16）**：
- NCI 0：仅支持 ENA（主网络接口，不支持 EFA）
- NCI 1-16：支持 efa-only，每个 400 Gbps EFA 带宽
- 总 EFA 带宽：**6400 Gbps**
- 总 ENA 带宽：最高 3870 Gbps（需配置 NCI 1-16 的 DeviceIndex=1 ENA 接口）

本文档采用**保存 IP 地址方案**（1个私有 IP，6400 Gbps EFA + 350 Gbps ENA）。

### 启动命令

```bash
aws ec2 run-instances \
  --region us-west-2 \
  --instance-type p6-b300.48xlarge \
  --image-id ami-06764d4b3d62da4fa \
  --count 2 \
  --key-name myp300 \
  --instance-market-options '{"MarketType":"capacity-block"}' \
  --capacity-reservation-specification '{"CapacityReservationTarget":{"CapacityReservationId":"cr-0d34cf36b15d769e0"}}' \
  --placement '{"AvailabilityZone":"us-west-2a","GroupName":"p6-efa-nccl-pg"}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=p6-b300-efa-nccl}]' \
  --network-interfaces \
    "NetworkCardIndex=0,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=interface,DeleteOnTermination=true" \
    "NetworkCardIndex=1,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=2,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=3,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=4,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=5,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=6,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=7,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=8,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=9,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=10,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=11,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=12,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=13,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=14,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=15,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true" \
    "NetworkCardIndex=16,DeviceIndex=0,Groups=sg-0957308fc5a969ef0,SubnetId=subnet-0ec7bbd671bb3a348,InterfaceType=efa-only,DeleteOnTermination=true"
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| `--instance-market-options` | `MarketType=capacity-block` | Capacity Block 必须指定此参数 |
| `--capacity-reservation-specification` | CR ID | 指定使用的 Capacity Block |
| `--placement GroupName` | p6-efa-nccl-pg | cluster 置放群组，保证低延迟 |
| NCI 0 InterfaceType | `interface` | 主网卡只能用 ENA，不支持 EFA |
| NCI 1-16 InterfaceType | `efa-only` | 16 个 EFA 接口，共 6400 Gbps |
| EBS VolumeType | `gp3` | 500 GiB gp3 |

---

## 启动结果

| 实例 ID | 私有 IP | 公有 EIP |
|---------|---------|---------|
| i-05bd8036eecb3ee84 | 172.31.35.107 | 35.164.97.227 |
| i-0cd02c2c4ac8baa76 | 172.31.47.177 | 35.81.195.174 |

- IAM Role：`b300test`（AdministratorAccess）
- 子网：`subnet-0ec7bbd671bb3a348`（us-west-2a，默认 VPC 公有子网）

---

## 步骤四：EFA + NCCL 软件安装（登录实例后执行）

### 4.1 安装 EFA 软件

```bash
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.47.0.tar.gz
tar -xf aws-efa-installer-1.47.0.tar.gz && cd aws-efa-installer
sudo ./efa_installer.sh -y --mpi=openmpi4
```

### 4.2 验证 EFA（应返回 16 个 efa 接口）

```bash
fi_info -p efa -t FI_EP_RDM
```

### 4.3 安装 NCCL

```bash
cd /opt
sudo git clone https://github.com/NVIDIA/nccl.git -b v2.23.4-1 && cd nccl
sudo make -j src.build CUDA_HOME=/usr/local/cuda
```

### 4.4 安装 NCCL 测试

```bash
cd $HOME
git clone https://github.com/NVIDIA/nccl-tests.git && cd nccl-tests
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
make MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME=/opt/nccl/build CUDA_HOME=/usr/local/cuda
```

### 4.5 配置无密码 SSH（节点间通信）

在节点1上生成密钥并分发到节点2：

```bash
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
ssh-copy-id ubuntu@172.31.47.177
```

### 4.6 运行 NCCL 跨节点测试

```bash
/opt/amazon/openmpi/bin/mpirun \
  -n 2 -N 1 \
  --hostfile my-hosts \
  -x FI_PROVIDER=efa \
  -x NCCL_DEBUG=INFO \
  --mca pml ^cm \
  $HOME/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 1
```

---

## 参考文档

- [EFA 加速器实例类型配置](https://docs.aws.amazon.com/zh_cn/AWSEC2/latest/UserGuide/efa-acc-inst-types.html)
- [EFA + NCCL 入门指南](https://docs.aws.amazon.com/zh_cn/AWSEC2/latest/UserGuide/efa-start-nccl.html)
- [Capacity Block 启动实例](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-launch.html)
