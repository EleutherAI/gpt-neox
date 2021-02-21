FROM atlanticcrypto/cuda-ssh-server:10.2-cudnn as build

SHELL [ "/bin/bash", "--login", "-c" ]

#### System package
RUN apt-get update -y && \
    apt-get install -y \
        git python3.8 python3.8-dev libpython3.8-dev  python3-pip python3-venv sudo pdsh \
        htop llvm-9-dev tmux zstd libpython3-dev software-properties-common build-essential autotools-dev \
        nfs-common pdsh cmake g++ gcc curl wget tmux less unzip htop iftop iotop ca-certificates \
        rsync iputils-ping net-tools llvm-9-dev libcupti-dev && \
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
    echo 'export PDSH_RCMD_TYPE=ssh' >> /home/mchorse/.bashrc && \
    chown -R mchorse:mchorse /home/mchorse

#### SWITCH TO mchorse USER
USER mchorse

# install miniconda
ENV CONDA_DIR ~/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# setup conda env
ENV CONDA_ENV megatron
RUN conda create --name $CONDA_ENV -y

# install torch from scratch
RUN conda install -n $CONDA_ENV numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
RUN conda install -n $CONDA_ENV -c pytorch magma-cuda102


SHELL ["/home/mchorse/miniconda3/bin/conda", "run", "--no-capture-output", "-n", "megatron", "/bin/bash", "-c"]

WORKDIR /home/mchorse
RUN git clone --recursive https://github.com/pytorch/pytorch && \
    cd pytorch && git submodule sync && git submodule update --init --recursive && \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && echo $CMAKE_PREFIX_PATH && python setup.py install && cd ..


#### Python packages
RUN python -m pip install --upgrade pip && \
    pip install gpustat

COPY requirements.txt $STAGE_DIR
RUN pip install -r $STAGE_DIR/requirements.txt
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git


################### END SLOW SECTION ###################
# WARNING: Avoid changing anything above this line, because it will take a *long* time to rebuild!


# this is a temporary patch because my /var/lib/docker has nosuid. remember to replace with just sudo later
USER root
RUN echo 'deb http://archive.ubuntu.com/ubuntu/ focal main restricted' > /etc/apt/sources.list && apt-get install --upgrade libpython3-dev

RUN apt-get install -y libpython3-dev

# Clear staging
RUN rm -r $STAGE_DIR /home/mchorse/pytorch && mkdir -p /tmp && chmod 0777 /tmp

# optimization so we don't have to push all of our layers
FROM scratch
COPY --from=build / /

USER mchorse
WORKDIR /home/mchorse

ENV PATH="/usr/local/mpi/bin:~/miniconda3/bin:/home/mchorse/.local/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}"

ENTRYPOINT ["/home/mchorse/miniconda3/bin/conda", "run", "--no-capture-output", "-n", "megatron", "python"]
