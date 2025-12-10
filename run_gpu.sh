#!/bin/bash
#
# ====== Slurm 配置 ======
#SBATCH --job-name=nn_train          # 任务名字
#SBATCH --partition=submit-gpu       # GPU 分区
#SBATCH --gres=gpu:1                 # 申请 1 块 GPU
#SBATCH --cpus-per-gpu=8             # 每块 GPU 对应 CPU 核心
#SBATCH --mem=60G                    # 内存
#SBATCH --time=08:00:00              # 最长运行时间
#SBATCH --constraint=nvidia_a30      # 指定 A30 GPU
#SBATCH --output=logs/%x-%j.out      # 标准输出日志
#SBATCH --error=logs/%x-%j.err       # 错误日志

# ====== 环境准备 ======

mkdir -p logs

cd /work/submit/haoyun22/FCC-Beam-Background/

# source /work/submit/jaeyserm/software/FCCAnalyses/setup.sh
source /work/submit/anton100/msci-project/smart-pixels-ml/venv/bin/activate
# 如果你还有自己的虚拟环境，这里可以激活，例如：
# source /work/submit/haoyun22/myenv/bin/activate

echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "CUDA available in Python:"
python - << 'EOF'
import torch
print("torch.__version__:", getattr(torch, "__version__", "no torch"))
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

# ====== 启动训练 ======
# 这里换成你真正的训练脚本和参数
echo "Start training..."
# python AB_Full_Classifier.py --neural
python CNN.py --classify --exp_name baseline


echo "Job finished."