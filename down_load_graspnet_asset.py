from huggingface_hub import snapshot_download
# snapshot_download(
#                 repo_id="SIMPLE-org/SIMPLE",
#                 allow_patterns=["assets.zip"],
#                 local_dir="/home/wyt/DexGrasp/BODex",
#                 repo_type="dataset",
#                 # resume_download=True,

#             )
import zipfile
import os
local_data_dir = "/home/wyt/DexGrasp/BODex"
zip_path = os.path.join(local_data_dir, "assets.zip")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(local_data_dir)