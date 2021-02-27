FROM atlanticcrypto/cuda-ssh-server:10.2-cudnn

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

ENV DEBIAN_FRONTEND=noninteractive

#### Temporary Installation Directory
ENV STAGE_DIR=/build
RUN mkdir -p ${STAGE_DIR}

#### OPENMPI
ENV OPENMPI_BASEVERSION=4.0
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.1
RUN cd ${STAGE_DIR} && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++ && \
    cd ${STAGE_DIR} && \
    rm -r ${STAGE_DIR}/openmpi-${OPENMPI_VERSION}
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}
# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun

#### User account
RUN useradd --create-home --uid 1000 --shell /bin/bash mchorse && \
    usermod -aG sudo mchorse && \
    echo "mchorse ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

## SSH config
RUN mkdir -p /home/mchorse/.ssh /job && \
    echo 'Host *' > /home/mchorse/.ssh/config && \
    echo '    StrictHostKeyChecking no' >> /home/mchorse/.ssh/config && \
    echo 'AuthorizedKeysFile     .ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    echo 'export PDSH_RCMD_TYPE=ssh' >> /home/mchorse/.bashrc

#### Python packages
RUN python -m pip install --upgrade pip && \
    pip install gpustat && \
    pip install torch==1.7.1

COPY requirements.txt $STAGE_DIR
RUN pip install -r $STAGE_DIR/requirements.txt
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git
RUN echo 'deb http://archive.ubuntu.com/ubuntu/ focal main restricted' >> /etc/apt/sources.list && apt-get install --upgrade libpython3-dev
RUN sudo apt-get update -y && sudo apt-get install -y libpython3-dev

# Clear staging
RUN rm -r $STAGE_DIR && mkdir -p /tmp && chmod 0777 /tmp

#### SWITCH TO mchorse USER
USER mchorse
WORKDIR /home/mchorse
ENV PATH="/home/mchorse/.local/bin:${PATH}"
