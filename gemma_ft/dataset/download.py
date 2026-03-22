"""
download.py — 從 kinetic_4K.jsonl 解析 YouTube ID 並擷取 frames

依賴：
  pip install yt-dlp
  brew install ffmpeg   # 或 apt install ffmpeg

用法：
  uv run python download.py                   # 下載全部（4478 支）
  uv run python download.py --limit 100       # 只下載前 100 個 clip（測試用）
  uv run python download.py --workers 4       # 並行 4 個進程

輸出結構（IMAGE_FOLDER = dataset/）：
  frames/
    5-xGskbsBgI_000055_000065/
      frame_1.jpg  frame_2.jpg  frame_3.jpg  frame_4.jpg
    ...
"""

import argparse
import json 
import os
import re
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ─────────────────────────────────────────────────────────────────────

JSONL_PATH   = Path(__file__).parent / "kinetic_4K.jsonl"
OUTPUT_DIR   = Path(__file__).parent / "frames"   # IMAGE_FOLDER/frames/
NUM_FRAMES   = 4      # frames per clip
TMP_DIR      = Path("/tmp/kinetics_videos")

# ── Parse clips from JSONL ─────────────────────────────────────────────────────

def parse_clips(jsonl_path: Path) -> list[dict]:
    """
    Returns list of:
      {"yt_id": "5-xGskbsBgI", "start": 55, "end": 65, "folder": "5-xGskbsBgI_000055_000065"}
    """
    clips = {}  # deduplicate by folder name
    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            for msg in sample["messages"]:
                for item in msg.get("content", []):
                    if item.get("type") != "image":
                        continue
                    # "frames/5-xGskbsBgI_000055_000065/frame_1.jpg"
                    parts = item["image"].split("/")
                    folder = parts[1]   # "5-xGskbsBgI_000055_000065"
                    if folder in clips:
                        continue
                    m = re.match(r"^(.+)_(\d{6})_(\d{6})$", folder)
                    if not m:
                        continue
                    yt_id = m.group(1)
                    start = int(m.group(2))
                    end   = int(m.group(3))
                    clips[folder] = {"yt_id": yt_id, "start": start, "end": end, "folder": folder}
    return list(clips.values())


# ── Download + extract frames ──────────────────────────────────────────────────

def download_clip(clip: dict) -> tuple[str, bool, str]:
    """
    1. yt-dlp: download only the needed time range → tmp mp4
    2. ffmpeg: extract NUM_FRAMES equally-spaced frames → OUTPUT_DIR/folder/frame_*.jpg
    Returns (folder, success, error_msg)
    """
    folder_name = clip["folder"]
    out_dir     = OUTPUT_DIR / folder_name

    # Skip if already done
    if all((out_dir / f"frame_{i}.jpg").exists() for i in range(1, NUM_FRAMES + 1)):
        return folder_name, True, "already exists"

    out_dir.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    yt_id  = clip["yt_id"]
    start  = clip["start"]
    end    = clip["end"]
    url    = f"https://www.youtube.com/watch?v={yt_id}"
    tmp_mp4 = TMP_DIR / f"{folder_name}.mp4"

    # ── Step 1: Download ──
    dl_cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "-f", "mp4/bestvideo[height<=480]+bestaudio/best",  # 最多 480p 節省空間
        "--download-sections", f"*{start}-{end}",           # 只下載需要的秒數
        "--force-keyframes-at-cuts",
        "-o", str(tmp_mp4),
        url,
    ]

    try:
        result = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or not tmp_mp4.exists():
            return folder_name, False, f"yt-dlp failed: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return folder_name, False, "yt-dlp timeout"

    # ── Step 2: Extract frames ──
    # 取得下載下來的影片真實長度（以防影片太短，或是 cut 不精準）
    probe_cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", str(tmp_mp4)
    ]
    try:
        actual_duration_str = subprocess.check_output(probe_cmd, text=True).strip()
        actual_duration = float(actual_duration_str)
    except Exception:
        actual_duration = end - start  # fallback

    # 如果影片小於 0.1 秒，或者有問題
    if actual_duration <= 0.1:
        actual_duration = end - start

    timestamps = [actual_duration * (i + 1) / (NUM_FRAMES + 1) for i in range(NUM_FRAMES)]

    for i, ts in enumerate(timestamps, start=1):
        out_frame = out_dir / f"frame_{i}.jpg"
        ff_cmd = [
            "ffmpeg",
            "-ss", str(ts),
            "-i", str(tmp_mp4),
            "-vframes", "1",
            "-q:v", "2",               # JPEG quality (2 = high)
            "-vf", "scale=336:336",    # Gemma 3 image size
            str(out_frame),
            "-y",                      # overwrite
        ]
        try:
            subprocess.run(ff_cmd, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            return folder_name, False, f"ffmpeg timeout at frame {i}"

    # Cleanup tmp file
    try:
        tmp_mp4.unlink()
    except OSError:
        pass

    # Verify output
    missing = [f"frame_{i}.jpg" for i in range(1, NUM_FRAMES + 1)
               if not (out_dir / f"frame_{i}.jpg").exists()]
    if missing:
        return folder_name, False, f"missing frames: {missing}"

    return folder_name, True, ""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download Kinetics-4K clips and extract frames")
    parser.add_argument("--limit",  type=int, default=None, help="Max clips to process (for testing)")
    parser.add_argument("--workers", type=int, default=4,    help="Parallel download workers")
    args = parser.parse_args()

    clips = parse_clips(JSONL_PATH)
    if args.limit:
        clips = clips[:args.limit]

    total = len(clips)
    print(f"Clips to process: {total}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Workers: {args.workers}")
    print()

    done = skipped = failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_clip, clip): clip for clip in clips}
        for i, fut in enumerate(as_completed(futures), start=1):
            folder, ok, msg = fut.result()
            if ok:
                if msg == "already exists":
                    skipped += 1
                else:
                    done += 1
                status = "✓" if msg != "already exists" else "="
            else:
                failed += 1
                status = "✗"
            print(f"[{i:4d}/{total}] {status} {folder[:50]:<50}  {msg[:60] if not ok else ''}")

    print()
    print(f"Done: {done}  Skipped: {skipped}  Failed: {failed}")
    if failed > 0:
        print("Re-run the script to retry failed downloads.")


if __name__ == "__main__":
    main()
