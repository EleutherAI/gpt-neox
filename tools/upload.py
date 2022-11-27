import os

from huggingface_hub import HfApi, create_repo

converted_ckpt = input("Where is the checkpoint folder (HF format) you want to use? ")
repo_name = input("Provide a repository name for the HF Hub: ")
branch_name = input("Provide a repo branch for the HF Hub (choose main as default): ")
create_repo(repo_name, repo_type="model", private=False)

files = os.listdir(converted_ckpt)

api = HfApi()
if branch_name != "main":
    api.create_branch(
        repo_id=repo_name,
        repo_type="model",
        branch=branch_name,
    )
print(f"to upload: {files}")
for file in files:
    print(f"Uploading {file}...")
    api.upload_file(
        path_or_fileobj=os.path.join(converted_ckpt, file),
        path_in_repo=file,
        repo_id=repo_name,
        repo_type="model",
        commit_message=f"Upload {file}",
        revision=branch_name,
    )
    print(f"Successfully uploaded {file} !")

