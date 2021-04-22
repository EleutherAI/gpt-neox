import subprocess
from dataclasses import dataclass

def get_git_commit_hash():
    """ Gets the git commit hash of your current repo (if it exists) """
    try:
        git_hash = subprocess.check_output(["git", "describe", "--always"]).strip()
        git_hash = git_hash.decode()
    except subprocess.CalledProcessError:
        git_hash = None
    return git_hash

@dataclass
class NeoXArgsLogging:

    wandb_group: str = None
    """Weights and Biases group name - used to group together "runs"."""

    wandb_team: str = None
    """Team name for Weights and Biases."""

    git_hash: str = get_git_commit_hash()
    """current git hash of repository"""

    log_dir: str = None
    """
    Directory to save logs to.
    """

    tensorboard_dir: str = None
    """
    Write TensorBoard logs to this directory.
    """
