#!/bin/bash
#
# ====== Slurm 配置 ======
#SBATCH --job-name=CNN_2c_dphi_ec002        # 任务名字
#SBATCH --partition=submit-gpu       # GPU 分区
#SBATCH --gres=gpu:1                 # 申请 1 块 GPU
#SBATCH --cpus-per-gpu=4             # 每块 GPU 对应 CPU 核心
#SBATCH --mem=24G                    # 内存
#SBATCH --time=4:00:00              # 最长运行时间
#SBATCH --constraint=nvidia_a30      # 指定 A30 GPU
#SBATCH --output=logs/%x-%j.out      # 标准输出日志
#SBATCH --error=logs/%x-%j.err       # 错误日志

# ====== 环境准备 ======

mkdir -p logs

cd /work/submit/haoyun22/FCC-Beam-Background/

# 清理可能干扰虚拟环境的环境变量
unset PYTHONPATH
unset PYTHONHOME

# 注释掉FCCAnalyses setup，避免numpy版本冲突
# source /work/submit/jaeyserm/software/FCCAnalyses/setup.sh
source /work/submit/haoyun22/FCC-Beam-Background/FCC310/bin/activate

# 打印Python路径，确认使用正确的虚拟环境
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# 设置CUDA环境变量
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "Checking CUDA availability:"
python - << 'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
else:
    print("WARNING: CUDA not available, will use CPU!")
EOF

# ====== 启动训练 ======
# 这里换成你真正的训练脚本和参数
echo "Start training..."
# python AB_Full_Classifier.py --test
python /work/submit/haoyun22/FCC-Beam-Background/CNN_AB_new/CNN.py --classify --exp_name CNN_2c_25um_dphi_ec002 --data /ceph/submit/data/user/h/haoyun22/CNN_data/AB_patches_2c_25um_dphi_ec002.npz
echo "Training script finished."

echo "Start evaluation..."
python /work/submit/haoyun22/FCC-Beam-Background/CNN_AB_new/CNN.py --evaluate --exp_name CNN_2c_25um_dphi_ec002
echo "Evaluation script finished."

# python train_epm_ddpm.py
# python /work/submit/haoyun22/FCC-Beam-Background/diffusion_model/compare.py
echo "Job finished."