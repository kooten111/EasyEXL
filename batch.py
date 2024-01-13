import sys
import subprocess
import threading
import json
import os
from huggingface_hub import HfApi, create_repo

def run_script(path, bpw_values, upload):
    absolute_path = os.path.abspath(path)

    folder_name = os.path.basename(os.path.normpath(absolute_path))
    for bpw in bpw_values:
        command = ["python", "EasyEXL.py", absolute_path, "--bpw", str(bpw)]
        subprocess.run(command)

    if upload:
        upload_thread = threading.Thread(target=upload_models, args=(folder_name, bpw_values, absolute_path))
        upload_thread.start()

def upload_models(folder_name, bpw_values, base_path):
    with open("settings.json", "r") as file:
        settings = json.load(file)
    userhf = settings.get("userhf", "")
    api = HfApi()

    for bpw in bpw_values:
        repo_name = f"{userhf}/{folder_name}-{bpw}bpw-exl2".lstrip('/')
        create_repo(repo_name, private=True)
        model_folder_path = os.path.join(base_path, f"{folder_name}-{bpw}bpw-exl2")
        api.upload_folder(
            folder_path=model_folder_path,
            repo_id=repo_name,
            repo_type="model",
        )

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python batch.py /path/to/model/ --bpw "8,6,5,4" [--upload]')
        sys.exit(1)

    model_path = sys.argv[1]
    bpw_arg_index = sys.argv.index("--bpw") + 1 if "--bpw" in sys.argv else None
    upload = "--upload" in sys.argv
    if bpw_arg_index is None or bpw_arg_index >= len(sys.argv):
        print("Invalid or missing bpw argument")
        sys.exit(1)

    bpw_values = [value.strip() for value in sys.argv[bpw_arg_index].split(',') if value.strip()]
    print("bpw_values:", bpw_values)

    run_script(model_path, bpw_values, upload)
