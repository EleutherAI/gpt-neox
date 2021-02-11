FROM atlanticcrypto/cuda-ssh-server:10.2-cudnn

RUN apt-get update && \
    apt-get install -y git python3.8 python3.8-dev libpython3.8-dev libopenmpi-dev python3-pip python3-venv sudo pdsh htop llvm-9-dev cmake tmux zstd && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    sudo apt-get update -y && sudo apt-get install -y libpython3-dev \
    python3 -m pip install --upgrade pip && \
    pip3 install pipx gpustat && \
    python3 -m pipx ensurepath

RUN mkdir -p ~/.ssh /app /job /build_dir && \
    echo 'Host *' > ~/.ssh/config && \
    echo '    StrictHostKeyChecking no' >> ~/.ssh/config && \
    echo 'AuthorizedKeysFile     .ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    echo 'export PDSH_RCMD_TYPE=ssh' >> ~/.bashrc

WORKDIR /build_dir

COPY requirements.txt /build_dir
RUN pip install torch==1.7.1
RUN pip install --upgrade tensorflow
RUN pip install -r requirements.txt
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git
RUN echo 'deb http://archive.ubuntu.com/ubuntu/ focal main restricted' >> /etc/apt/sources.list && apt-get install --upgrade libpython3-dev

WORKDIR /app

