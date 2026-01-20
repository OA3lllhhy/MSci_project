#!/bin/bash
#
# ====== Slurm 配置 ======
#SBATCH --job-name=CNN_2c_p        # 任务名字
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

source /work/submit/jaeyserm/software/FCCAnalyses/setup.sh
# source /work/submit/haoyun22/FCC-Beam-Background/FCC/bin/activate
# 如果你还有自己的虚拟环境，这里可以激活，例如：
# source /work/submit/haoyun22/myenv/bin/activate

echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
# echo "CUDA available in Python:"
# python - << 'EOF'
# import torch
# print("torch.__version__:", getattr(torch, "__version__", "no torch"))
# print("torch.cuda.is_available():", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))
# EOF

# ====== 启动训练 ======
# 这里换成你真正的训练脚本和参数
echo "Start training..."
# python AB_Full_Classifier.py --test
python /work/submit/haoyun22/FCC-Beam-Background/CNN_AB_new/CNN.py --classify --exp_name CNN_2c_p64 --data /ceph/submit/data/user/h/haoyun22/CNN_data/AB_patches_2c_64.npz
echo "Training script finished."

echo "Start evaluation..."
python /work/submit/haoyun22/FCC-Beam-Background/CNN_AB_new/CNN.py --evaluate --exp_name CNN_2c_p64 --data /ceph/submit/data/user/h/haoyun22/CNN_data/AB_patches_2c_64.npz
echo "Evaluation script finished."

# python train_epm_ddpm.py
# python /work/submit/haoyun22/FCC-Beam-Background/diffusion_model/compare.py
# echo "Job finished."