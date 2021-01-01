sudo apt-get -y install llvm-9-dev cmake
git clone https://github.com/microsoft/DeepSpeed.git /tmp/Deepspeed
cd /tmp/Deepspeed && DS_BUILD_SPARSE_ATTN=1 ./install.sh -s
