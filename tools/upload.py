# Copyright (c) 2021, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
