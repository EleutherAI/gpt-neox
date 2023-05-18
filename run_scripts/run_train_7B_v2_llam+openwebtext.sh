#!/bin/bash

### 将本次作业计费到导师课题组，tutor_project改为导师创建的课题组名
#SBATCH --comment=joint_project

### 给你这个作业起个名字，方便识别不同的作业
#SBATCH --job-name=train_1-3B_v7-llama7b+linly

### 指定该作业需要多少个节点
### 注意！没有使用多机并行（MPI/NCCL等），下面参数写1！不要多写，多写了也不会加速程序！
#SBATCH --nodes=8

### 指定该作业需要多少个CPU核心
### 注意！一般根据队列的CPU核心数填写，比如cpu队列64核，这里申请64核，并在你的程序中尽量使用多线程充分利用64核资源！
#SBATCH --gres=gpu:8

### 指定该作业在哪个队列上执行
#SBATCH --partition=gpu-a800-gsai

### 以上参数用来申请所需资源
### 以下命令将在计算节点执行

### 本例使用Anaconda中的Python，先将Python添加到环境变量配置好环境变量
# source activate llm
nodelist=$(scontrol show hostname $SLURM_NODELIST)
for i in "${nodelist[@]}"
do
   printf "%s slots=8\n" $i > 7B_v2_hostfile
   # or do whatever with individual element of the array
done

### 执行你的作业
module load cuda/11.8
source activate llm
python ./deepy.py train.py -d jarvis_configs/7B/v2-llama+open_web_text params.yml setup.yml -H 7B_v2_hostfile

rm 7B_v2_hostfile
