FROM nvcr.io/nvidia/pytorch:21.02-py3

ENV DEBIAN_FRONTEND=noninteractive

#### System package
RUN apt-get update -y && \
    apt-get install -y \
        git python3.8 python3.8-dev libpython3.8-dev  python3-pip python3-venv sudo pdsh \
        htop llvm-9-dev tmux zstd libpython3-dev software-properties-common build-essential autotools-dev \
        nfs-common pdsh cmake g++ gcc curl wget tmux less unzip htop iftop iotop ca-certificates \
        rsync iputils-ping net-tools libcupti-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

#### Temporary Installation Directory
ENV STAGE_DIR=/build
RUN mkdir -p ${STAGE_DIR}

#### User account
RUN useradd --create-home --uid 1000 --shell /bin/bash mchorse && \
    usermod -aG sudo mchorse && \
    echo "mchorse ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

## SSH config and bashrc
RUN mkdir -p /home/mchorse/.ssh /job && \
    echo 'Host *' > /home/mchorse/.ssh/config && \
    echo '    StrictHostKeyChecking no' >> /home/mchorse/.ssh/config && \
    echo 'AuthorizedKeysFile     .ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    echo 'export PDSH_RCMD_TYPE=ssh' >> /home/mchorse/.bashrc && \
    echo 'export PATH=/home/mchorse/.local/bin:$PATH' >> /home/mchorse/.bashrc && \
    echo 'export PATH=/usr/local/mpi/bin:$PATH' >> /home/mchorse/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH' >> /home/mchorse/.bashrc

#### Python packages
RUN python -m pip install --upgrade pip
RUN pip install pybind11==2.6.2 six regex nltk==3.5 zstandard==0.15.1 cupy-cuda112==8.4.0 mpi4py==3.0.3 wandb==0.10.18 einops==0.3.0 gpustat
RUN pip install -e git+git://github.com/EleutherAI/DeeperSpeed.git@cac19a86b67e6e98b9dca37128bc01e50424d9e9#egg=deepspeed
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git@e2083df5eb96643c61613b9df48dd4eea6b07690
RUN echo 'deb http://archive.ubuntu.com/ubuntu/ focal main restricted' >> /etc/apt/sources.list && apt-get install --upgrade libpython3-dev
RUN sudo apt-get update -y && sudo apt-get install -y libpython3-dev

# Clear staging
RUN rm -r $STAGE_DIR && mkdir -p /tmp && chmod 0777 /tmp

#### SWITCH TO mchorse USER
USER mchorse
WORKDIR /home/mchorse
ENV PATH="/home/mchorse/.local/bin:${PATH}"
