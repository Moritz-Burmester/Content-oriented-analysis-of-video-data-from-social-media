import csv
import glob
import sys
import os
import pandas as pd

#TODO: How to have all imports active at the same time and not throw erros with different environments
#from classification_videollava import init_videoll aava, classify_videollava
#from classification_pandagpt import init_pandagpt, classify_pandagpt
from classification_videochatgpt import init_videochatgpt, classify_videochatgpt

"""
For accessing the different models

Use the corresponding conda environment for the specific model
"""

dataset_path = "/ceph/lprasse/ClimateVisions/Videos"
duplicates = "/work/mburmest/bachelorarbeit/Duplicates/duplicates.csv"
env_name = os.environ.get("CONDA_DEFAULT_ENV")
solution_path = os.path.join("/work/mburmest/bachelorarbeit", env_name + "_solution.csv")

# Before starting the program select a Prompt
prompt = "What is the main color in the video?"

# Before starting the program select a conda environment
def main():
  set_duplicates = load_csv_into_set(duplicates)
  videos = glob.glob(f"{dataset_path}/*/*/*.mp4")
  print("Selected: " + env_name)

  if env_name == "videollava":
    video_processor, tokenizer, model = init_videollava()
  elif env_name == "pandagpt":
    model, max_length, top_p, temperature = init_pandagpt()
  elif env_name == "videochatgpt":
    model, model_name, vision_tower, tokenizer, image_processor, video_token_len, temperature, max_output_tokens = init_videochatgpt()
  else:
    print("Error: Cannot find the selected model")
    sys.exit(1)

  
  for video in videos:
    id_string = video.split("/")[-1].split(".")[0]

    if checkDuplicate(id_string,set_duplicates):
      print("\nDuplicate eliminated\n")
      continue
    
    if not os.path.exists(solution_path):
      df = pd.DataFrame(columns=["id", "solution"])
      df.to_csv(solution_path, index=False)
    
    #TODO: Frame by Frame model
    if env_name == "videollava":
      result = classify_videollava(video, prompt, model , video_processor, tokenizer)
      print(result)
    elif env_name == "pandagpt":
      result = classify_pandagpt(video, prompt, model, max_length, top_p, temperature)
      print(result)
    elif env_name == "videochatgpt":
      result = classify_videochatgpt(video, prompt, model, model_name, vision_tower, tokenizer, image_processor, video_token_len, temperature, max_output_tokens)
      print(result)
    else:
      print("Error: Cannot find the selected model")
      break
       
    with open(solution_path, mode="a", newline="") as solution:
      writer = csv.writer(solution)
      writer.writerow([id_string, result])

  
  #Sort df
  df = pd.read_csv(solution_path)
  df["date"] = pd.to_datetime(df["id"].str.split("_").str[-1])
  df = df.sort_values(by="date")
  df.drop(columns=["date"]).to_csv(solution_path, index = False) 

# Load Duplicate file into a set for fast lookups
def load_csv_into_set(file_path: str) -> set:
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            return {cell for row in reader for cell in row}  # Flattening rows into a set
    except:
        print("Error in load_csv_into_set()")
        sys.exit(1)

# Check if the video is in the duplicate file
# @param video_id: "id_x...x_YYYY-MM-DD"
# @param dup_set: Set with all duplicate ids created by load_csv_into_set()
# @return: Returns true if duplicate is present, returns false in every other case
def checkDuplicate(video_id : str, dup_set : set) -> bool:
    return video_id in dup_set
    
if __name__ == "__main__": 
  main()