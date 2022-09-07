```bash
# conda setup
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow instructions, accept all conditions blindly

# Modify This:
export CONDA_ENV_PATH=/fsx/conda/env/path
export CODE_BASE_PATH=/fsx/code/path

# Create env
conda create --prefix ${CONDA_ENV_PATH} python=3.9
conda activate ${CONDA_ENV_PATH}

# Install a bunch of things 
conda install -y pytorch torchvision torchaudio cudatoolkit=11.4 -c pytorch -c conda-forge
conda install -y jupyter notebook tqdm matplotlib pandas scikit-learn
pip install transformers datasets wandb bitsandbytes accelerate rouge-score absl-py nltk scipy sklearn deepspeed tokenizers

# Gotta build mpi4py for openmpi
cd ${CODE_BASE_PATH}
pip uninstall mpi4py
get checkout -b stability git@github.com:zphang/mpi4py.git
cd mpi4py
# Don't let ld get in the way
mv ${CONDA_ENV_PATH}/compiler_compat/ld ${CONDA_ENV_PATH}/compiler_compat/ld_backup
python setup.py clean --all
python setup.py build --mpi=openmpi
python setup.py install
mv ${CONDA_ENV_PATH}/compiler_compat/ld_backup ${CONDA_ENV_PATH}/compiler_compat/ld

# Install gpt-neox, deeperspeed, etc
cd ${CODE_BASE_PATH}
git clone git@github.com:EleutherAI/gpt-neox.git
cd gpt-neox
# Modify the requirements.txt in neox to take out the mpi4py requirement
pip install -r requirements/requirements.txt
git clone -b stability git@github.com:EleutherAI/DeeperSpeed.git
```