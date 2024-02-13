# Copyright (c) 2024, EleutherAI
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
import sys

from huggingface_hub import HfApi, create_repo

converted_ckpt = sys.argv[1]
repo_name = sys.argv[2]
branch_name = sys.argv[3]
try:
    create_repo(repo_name, repo_type="model", private=False)
except:
    print("repo {repo_name} already exists!")
    pass

files = os.listdir(converted_ckpt)

api = HfApi()
if branch_name != "main":
    try:
        api.create_branch(
            repo_id=repo_name,
            repo_type="model",
            branch=branch_name,
        )
    except:
        print(f"branch {branch_name} already exists, try again...")
print(f"to upload: {files}")
for file in files:
    print(f"Uploading {file} to branch {branch_name}...")
    api.upload_file(
        path_or_fileobj=os.path.join(converted_ckpt, file),
        path_in_repo=file,
        repo_id=repo_name,
        repo_type="model",
        commit_message=f"Upload {file}",
        revision=branch_name,
    )
    print(f"Successfully uploaded {file} !")
