FROM atlanticcrypto/cuda-ssh-server:10.2-cudnn

RUN apt-get update && \
    apt-get install -y git python3.8 python3.8-dev python3-pip sudo pdsh htop llvm-9-dev cmake tmux && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
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
RUN pip install -r requirements.txt

WORKDIR /app
