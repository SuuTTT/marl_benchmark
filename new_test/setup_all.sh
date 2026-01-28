#!/bin/bash
# Fast setup script for new Vast.ai instances

echo "🚀 Starting Automated Environment Setup..."

# Update and install system dependencies for VMAS/OpenGL
apt-get update && apt-get install -y libgl1-mesa-glx libosmesa6-dev python3-pip

# Install Python ML stack
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install benchmarl vmas torchrl wandb gymnasium tensorboard -q

# Verify Installation
python3 -c "import torch; import benchmarl; import vmas; print('✅ Setup Complete! GPU:', torch.cuda.get_device_name(0))"
