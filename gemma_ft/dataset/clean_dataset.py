import json
from pathlib import Path

# ── Config ──
JSONL_PATH = Path(__file__).parent / "kinetic_4K.jsonl"
CLEAN_JSON_PATH = Path(__file__).parent / "kinetic_4K_clean.json"
FRAMES_DIR = Path(__file__).parent / "frames"
NUM_FRAMES = 4

def main():
    print(f"Reading original dataset: {JSONL_PATH}")
    clean_samples = []
    missing_count = 0
    
    with open(JSONL_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            
            # Find the image folder for this sample
            is_valid = True
            for msg in sample.get("messages", []):
                for item in msg.get("content", []):
                    if item.get("type") == "image":
                        # item["image"] is like "frames/5-xGskbsBgI_000055_000065/frame_1.jpg"
                        # We just need the folder name
                        parts = item["image"].split("/")
                        if len(parts) >= 2:
                            folder = parts[1]
                            folder_path = FRAMES_DIR / folder
                            
                            # Check if all 4 frames exist
                            for i in range(1, NUM_FRAMES + 1):
                                if not (folder_path / f"frame_{i}.jpg").exists():
                                    is_valid = False
                                    break
            
            if is_valid:
                clean_samples.append(sample)
            else:
                missing_count += 1
                
    # Save the clean JSON list (Transformers Trainer expects JSON list, or JSONL)
    print(f"Writing clean dataset with {len(clean_samples)} valid clips...")
    print(f"Skipped {missing_count} clips due to missing frames or deleted YouTube videos.")
    
    with open(CLEAN_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_samples, f, indent=2, ensure_ascii=False)
        
    print(f"Done! Clean dataset saved to: {CLEAN_JSON_PATH}")

if __name__ == "__main__":
    main()
