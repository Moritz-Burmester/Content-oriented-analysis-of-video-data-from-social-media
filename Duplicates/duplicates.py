import cv2
import os
import hashlib
import json
from collections import defaultdict

# Path to dataset folder
DATASET_PATH = "/ceph/lprasse/ClimateVisions/Videos"

# Output files
DUPLICATE_IDS_FILE = "duplicate_ids.json"
ORIGINAL_TO_DUPLICATES_FILE = "original_to_duplicates.json"
VIDEO_HASHES_FILE = "video_hashes.json"  # New file for video hashes

# Function to compute hash of a frame
def hash_frame(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()

# Function to check if a frame is black
def is_black_frame(frame, threshold=10):
    return frame.mean() < threshold

# Function to recursively get all video files in dataset
def get_all_videos(root_folder, video_extensions=(".mp4", ".avi", ".mov", ".mkv")):
    video_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(dirpath, file))
    return video_files

# Step 1: Extract first non-black frame and hash it
video_files = get_all_videos(DATASET_PATH)
video_hashes = {}
hash_to_videos = defaultdict(list)
analyzed_count = 0  # Track analyzed videos

# Initialize counter for progress
total_videos = len(video_files)
print(f"Starting analysis of {total_videos} videos...")

for idx, video_path in enumerate(video_files, start=1):
    video_id = os.path.splitext(os.path.basename(video_path))[0]  # Extract video ID from filename
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open {video_id}")
        continue

    analyzed_count += 1
    first_hash = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if not is_black_frame(frame):  # First non-black frame
            first_hash = hash_frame(frame)
            video_hashes[video_id] = first_hash
            hash_to_videos[first_hash].append(video_id)
            break

    cap.release()

    # Print progress update every 100 videos and show percentage
    if idx % 100 == 0 or idx == total_videos:
        percentage = (idx / total_videos) * 100
        print(f"Processed {idx}/{total_videos} videos ({percentage:.2f}%)...")

# Step 2: Identify duplicates
duplicate_ids = set()
original_to_duplicates = {}

for hash_value, videos in hash_to_videos.items():
    if len(videos) > 1:
        original = videos[0]  # First occurrence is the original
        duplicates = videos[1:]  # Remaining videos are duplicates

        duplicate_ids.update(duplicates)
        original_to_duplicates[original] = duplicates

# Step 3: Save results to JSON files
with open(DUPLICATE_IDS_FILE, "w") as f:
    json.dump(list(duplicate_ids), f, indent=4)

with open(ORIGINAL_TO_DUPLICATES_FILE, "w") as f:
    json.dump(original_to_duplicates, f, indent=4)

# Step 4: Save video hashes to a JSON file
with open(VIDEO_HASHES_FILE, "w") as f:
    json.dump(video_hashes, f, indent=4)

# Print final summary
print(f"Duplicate check completed! {analyzed_count} videos were analyzed.")
print(f"Found {len(duplicate_ids)} duplicate videos.")
