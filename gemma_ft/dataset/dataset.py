# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
# ]
# ///
from datasets import load_dataset

dataset = load_dataset("lmms-lab/VATEX")

print("Dataset loaded successfully.")
print(dataset)

# 將 test 拆分的資料存成本機檔案 (CSV 與 JSON)
dataset["test"].to_csv("vatex_test.csv", index=False)
dataset["test"].to_json("vatex_test.json")

print("\n已經將資料存成本機檔案：vatex_test.csv 和 vatex_test.json")

# from huggingface_hub import snapshot_download

# # 只下載路徑或檔名包含 "kinetics" 的檔案（忽略大小寫需依實際檔名而定）
# downloaded_path = snapshot_download(
#     repo_id="TimingYang/ViMix-14M",
#     repo_type="dataset",
#     allow_patterns="*kinetics*",  # 關鍵：利用萬用字元過濾特定子集
#     local_dir="./ViMix_Kinetics700"
# )

# print(f"檔案已下載至：{downloaded_path}")