# Setting up your workspace on Stability

This guide walks you through how to set up conda environments, install libraries, etc until you are able to run `gpt-neox` and train on multiple nodes. This guide will assume you know the standard things (SSHing in, editing files server-side).

First (after SSHing in), let's set up conda. The easiest way to do so is to just install a base conda installation in your home directory.

```bash
# conda setup
cd /fsx/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow instructions, accept all conditions blindly
ln -s /fsx/zphang/setup/miniconda3 /home/zphang/miniconda3
ln -s /fsx/zphang/setup/miniconda3 /home/zphang/miniconda3
```

After that, close the SSH session, and SSH back in. Your bash should now be configured to run conda. You can run `which conda` to check that it's pointing to the right one.

Next, we will start setting up your conda environment. You may want to use different ones for different projects. (I'm actually not sure if the home directory or `/fsx` is a better place for putting the conda env.) You should be able to run the following, and accept all prompts.

```bash
# Modify This:
export CONDA_ENV_PATH=/fsx/conda/env/path
export CODE_BASE_PATH=/fsx/code/path

# Create env
conda create --prefix ${CONDA_ENV_PATH} python=3.9
conda activate ${CONDA_ENV_PATH}

# Install a bunch of things 
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y jupyter notebook tqdm matplotlib pandas scikit-learn
pip install transformers datasets wandb bitsandbytes accelerate rouge-score absl-py nltk scipy sklearn deepspeed tokenizers
```

Now we need to install a couple more involved libraries. First, we need to install a version of `mpi4py` that's built against the `openmpi` used on the cluster. I've already preconfigured it at `zphang/mpi4py` so you can just clone and install.

```bash
# Gotta build mpi4py for openmpi
cd ${CODE_BASE_PATH}
pip uninstall mpi4py
git clone --branch stability git@github.com:zphang/mpi4py.git
cd mpi4py
# Don't let ld get in the way. I don't know why but it interferes.
mv ${CONDA_ENV_PATH}/compiler_compat/ld ${CONDA_ENV_PATH}/compiler_compat/ld_backup
python setup.py clean --all
python setup.py build --mpi=openmpi
python setup.py install
# Put ld back
mv ${CONDA_ENV_PATH}/compiler_compat/ld_backup ${CONDA_ENV_PATH}/compiler_compat/ld
````

Next, we'll install `gpt-neox`, and the `DeeperSpeed` fork of `DeepSpeed`. **Note the part about removing the mpi4py requirement.**

```
# Install gpt-neox, deeperspeed, etc
cd ${CODE_BASE_PATH}
git clone git@github.com:EleutherAI/gpt-neox.git
cd gpt-neox
# NOTE: Modify the requirements.txt in neox to take out the mpi4py requirement
pip install -r requirements/requirements.txt
git clone -b stability git@github.com:EleutherAI/DeeperSpeed.git
```
