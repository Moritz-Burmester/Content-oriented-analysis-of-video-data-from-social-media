import glob
import hashlib
import csv
import os
import cv2

"""
This code caculates the hash-values of the first and last non-black frame of a video and inputs it in a csv file.
"""

dataset_path = "/ceph/lprasse/ClimateVisions/Videos"
# Output file
video_hash_values_db = "/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/video_hash_values.csv"

# Gets all the video files (mp4) in the structure path/year/month/id.mp4
files = glob.glob(f"{dataset_path}/*/*/*.mp4")


def getHashValue(frame):
    """
    Calculates a SHA-256 hash value for a given video frame.

    Parameters:
        frame: Frame of a video

    Returns:
        str: A hexadecimal SHA-256 hash string of the frame.
    """
    return hashlib.sha256(frame.tobytes()).hexdigest()

def blackFrame(frame):
    """
    returns true if a frame is black
    => Frame == Black if the frame.mean() < 5

    Parameters:
        frame: Frame of a video

    Returns:
        bool
    """
    return frame.mean() < 5

def getFirstFrame(video):
    """
    Get the first frame that is not-black (if the whole video is black a black frame is returned).
    
    Parameters:
        video (str): Path to the video

    Returns:
        First non black frame
    """

    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Error: Cannot open the video:\n" + video)
        return None
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video
            break

        if not blackFrame(frame):
            cap.release()
            # Return first non-black frame
            return frame 
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    print("Only Black -- First Frame\n" + video)
    cap.release() 
    # Only black frames found
    return frame

def getLastFrame(video):
    """
    Get the last frame that is not-black (if the whole video is black a black frame is returned)
    
    Parameters:
        video (str): Path to the video

    Returns:
        Last non black frame
    """


    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Error: Cannot open the video:\n" + video)
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # To make the code faster we go from the back to the front of the video
    for frame_nr in range(total_frames - 1, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
        ret, frame = cap.read()
        if not ret:
            continue

        if not blackFrame(frame):
            cap.release()
            # Return first non-black frame
            return frame 
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    print("Only Black -- Last Frame\n" + video)
    cap.release()
    # Only black frames found
    return frame

def init_video_hash_values_csv():
   """
   Inits the csv file video_hash_values.csv File has to be created before.
   """

   header = ["id", "hash1", "hash2"]

   if os.path.exists(video_hash_values_db):
       if os.stat(video_hash_values_db).st_size == 0:
           with open(video_hash_values_db, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
       else: 
           print("Error: File not empty")
   else: 
       print("Error: CSV-File not found")
           
def append_video_hash_values(id_value, hash1_value, hash2_value):
    """
    Write values in video_hash_values_db.

    Parameters:
        id_value (str): ID of the video
        hash1_value (str): Hash value of the first frame
        hash2_value (str): Hash value of the last frame
    """

    with open(video_hash_values_db, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([id_value, hash1_value, hash2_value])

# Start of the code

init_video_hash_values_csv()

total_files = len(files)
for i, file in enumerate(files):
    id_string = file.split("/")[-1].split(".")[0]

    """ -- Only for adding new videos
    with open(video_hash_values_db, mode="r", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header row
        for row in reader:
            if row and row[0] == id_string:  # Check if ID matches the first column
                continue
    """

    frame1 = getFirstFrame(file)
    frame2 = getLastFrame(file)

    if frame1 is None or frame2 is None:
        print("Error: " + id_string)
    continue

    hash1 = getHashValue(frame1)
    hash2 = getHashValue(frame2)

    append_video_hash_values(id_string, hash1, hash2)

    percentage = ((i + 1) / total_files) * 100

    if (i + 1) % 100 == 0:
        print(f"Proccessed {i + 1} files out of {total_files} ({percentage:.2f}%)")

print("Succesfully caculated all required hash-values of the videos")