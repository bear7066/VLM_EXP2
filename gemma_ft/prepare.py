import os
from huggingface_hub import snapshot_download

REPO = "bear7011/gemma-3-4b-kinetics_3K"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "dataset") # __file__ is the path to the current file

def main():
    print(f"Downloading dataset {REPO} to {LOCAL_DIR} ...")
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        resume_download=True
    )
    print(f"Download complete! Dataset saved to: {LOCAL_DIR}")

if __name__ == "__main__":
    main()