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
import sys

from huggingface_hub import HfApi, create_repo

converted_ckpt = sys.argv[1] #input("Where is the checkpoint folder (HF format) you want to use? ")
repo_name = sys.argv[2] #input("Provide a repository name for the HF Hub: ")
branch_name = sys.argv[3] #input("Provide a repo branch for the HF Hub (choose main as default): ")
try:
    create_repo(repo_name, repo_type="model", private=True, use_auth_token="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")
except:
    pass

files = os.listdir(converted_ckpt)

api = HfApi()
if branch_name != "main":
    api.create_branch(
        repo_id=repo_name,
        repo_type="model",
        branch=branch_name,
        use_auth_token="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",
    )

files = [file for file in files if (file.endswith(".bin") or file.endswith(".idx"))]

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
        use_auth_token="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",
    )
    print(f"Successfully uploaded {file} !")
