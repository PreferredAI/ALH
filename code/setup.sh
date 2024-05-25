#!/bin/bash

# issue guides for mujoco 1.50pro:
# https://github.com/openai/mujoco-py/issues/96#issuecomment-346685411
# https://github.com/ethz-asl/reinmav-gym/issues/35#issuecomment-1222946797
# conda install -c conda-forge mesa-libgl-cos6-x86_64
# conda install -c conda-forge --force-reinstall glew mesalib==17.3.9 glfw3


conda create -n test_env python=3.9
conda activate test_env
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c conda-forge glew -y
conda install -c conda-forge mesalib -y
conda install -c menpo glfw3 -y
echo 'export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc
echo 'export CPATH=$CONDA_PREFIX/include' >> ~/.bashrc
source ~/.bashrc
pip install patchelf

mkdir "./logs/"
mkdir -p ~/.mujoco \
    && curl https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -o mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C ~/.mujoco \
    && rm mujoco.tar.gz \
    && curl https://www.roboti.us/file/mjkey.txt -o ~/.mujoco/mjkey.txt

pip install "cython<3"
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
set CUDA_LAUNCH_BLOCKING=1
pip install -r ./requirements.txt > ./logs/setup.log