#!/bin/bash
#
# ====== Slurm 配置 ======
#SBATCH --job-name=test_cuda          # 任务名字
#SBATCH --partition=submit-gpu        # GPU 分区
#SBATCH --gres=gpu:1                  # 申请 1 块 GPU
#SBATCH --cpus-per-gpu=2              # 每块 GPU 对应 CPU 核心
#SBATCH --mem=8G                      # 内存
#SBATCH --time=0:10:00                # 最长运行时间
#SBATCH --constraint=nvidia_a30       # 指定 A30 GPU
#SBATCH --output=logs/test_cuda-%j.out      # 标准输出日志
#SBATCH --error=logs/test_cuda-%j.err       # 错误日志

# ====== 环境准备 ======
mkdir -p logs
cd /work/submit/haoyun22/FCC-Beam-Background/

echo "====== System Information ======"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "Date: $(date)"
echo ""

echo "====== GPU Information (nvidia-smi) ======"
nvidia-smi
echo ""

echo "====== CUDA Environment Variables ======"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "====== Activating Python Environment ======"
source /work/submit/haoyun22/FCC-Beam-Background/FCC310/bin/activate
which python
python --version
echo ""

echo "====== PyTorch CUDA Test ======"
python << 'EOF'
import sys
import torch
import os

print("=" * 60)
print("System Information:")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("")

print("PyTorch Information:")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch file location: {torch.__file__}")
print("")

print("CUDA Information:")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version (compiled): {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    
    # Test GPU computation
    print("")
    print("Testing GPU computation:")
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = x @ y
    print(f"Matrix multiplication on GPU successful!")
    print(f"Result device: {z.device}")
else:
    print("WARNING: CUDA is NOT available!")
    print("")
    print("Troubleshooting information:")
    print(f"CUDA_HOME env var: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Check if CUDA libraries are available
    import ctypes.util
    cuda_lib = ctypes.util.find_library('cuda')
    cudart_lib = ctypes.util.find_library('cudart')
    print(f"libcuda found: {cuda_lib}")
    print(f"libcudart found: {cudart_lib}")

print("=" * 60)
EOF

echo ""
echo "====== Test Complete ======"
