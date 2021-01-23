FROM atlanticcrypto/cuda-ssh-server:10.2-cudnn

RUN apt-get update && \
    apt-get install -y git python3.8 python3.8-dev python3-pip sudo pdsh && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    python3 -m pip install --upgrade pip && \
    pip3 install torch pipx && \
    python3 -m pipx ensurepath

RUN mkdir -p ~/.ssh /app && \
    echo 'Host *' > ~/.ssh/config && \
    echo '    StrictHostKeyChecking no' >> ~/.ssh/config && \
    echo 'AuthorizedKeysFile     .ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config

WORKDIR /app

COPY install_deepspeed.sh /app
RUN sh ./install_deepspeed.sh

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app