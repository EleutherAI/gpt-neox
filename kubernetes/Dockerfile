FROM atlanticcrypto/cuda-ssh-server:10.2-cudnn

RUN echo 'AuthorizedKeysFile     .ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    apt-get update && \
    apt-get install -y git python3.8 python3.8-dev python3-pip sudo && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN mkdir -p ~/.ssh && \
    echo 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQChClBdh2UXpMhBV725nH1bMsGVaLmYAp8uMZFHtBqHN56sj/35jeenQTqr/Tov3u6xwqJK+rxegjgcPDZfuOSdsNnTMCLIacA/WqBMwW1mjdMc+zFOub7vQJHj4nmeF3pd4tSjt720ZLiX1ZsF5QrTIcnURAXT0/82SKIy2nqj18v9HCcIvBplexJU3SlVg+oWk/e5CsfnXvJMQH3VqJaeyrXIlaFgVOvVWSBY66Kc2H+g1RxLwe+BONyNanxSXxKHQwUXawBvXIyjekapB3HiNWLZZfZjNmqe2Ci+Y9PKn0CrkgXopIP7tVKn+UQ2fD3nSjaZyfRrmcFgXdotFKJh root@sshd-sid-79b6f9d7c6-mmw8x' > ~/.ssh/authorized_keys

RUN python3 -m pip install --upgrade pip && \
    pip3 install torch pipx && \
    python3 -m pipx ensurepath
