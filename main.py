import csv
import glob
import sys
import os
import traceback
import pandas as pd
import torch

"""
For accessing the different models

Use the corresponding conda environment for the specific model
"""

dataset_path = "/ceph/lprasse/ClimateVisions/Videos"
duplicates = "/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/duplicates.csv"
env_name = os.environ.get("CONDA_DEFAULT_ENV")
solution_path = os.path.join("/work/mburmest/bachelorarbeit/", env_name + "_solution.csv")
exception_path = os.path.join("work/mburmest/bachelorarbeit/", env_name + "_exception.csv")

# Before starting the program select a Prompt
prompt = "What is the main color in the video?"

# Before starting the program select a conda environment
def main():
  print("Selected: " + env_name)

  if env_name == "videollava":
    from classification_videollava import init_videollava, classify_videollava
    model, video_processor, tokenizer = init_videollava()
    classify_model(classify_videollava, model, video_processor, tokenizer)
  elif env_name == "pandagpt":
    from classification_pandagpt import init_pandagpt, classify_pandagpt
    model, max_length, top_p, temperature = init_pandagpt()
    classify_model(classify_pandagpt, model, max_length, top_p, temperature)
  elif env_name == "videochatgpt":
    from classification_videochatgpt import init_videochatgpt, classify_videochatgpt
    model, model_name, vision_tower, tokenizer, image_processor, video_token_len, temperature, max_output_tokens = init_videochatgpt()
    classify_model(classify_videochatgpt, model, model_name, vision_tower, tokenizer, image_processor, video_token_len, temperature, max_output_tokens)
  else:
    print("Error: Cannot find the selected model")
    sys.exit(1)
  
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


def classify_model(classify, *args):
  set_duplicates = load_csv_into_set(duplicates)
  videos = glob.glob(f"{dataset_path}/*/*/*.mp4")


  if not os.path.exists(solution_path):
    df = pd.DataFrame(columns=["id", "solution"])
    df.to_csv(solution_path, index=False)

  for video in videos:
    id_string = video.split("/")[-1].split(".")[0]
    print("Video: " + id_string)

    if checkDuplicate(id_string,set_duplicates):
      print("Duplicate eliminated\n")
      continue

    try:
      result = classify("/ceph/lprasse/ClimateVisions/Videos/2021/11_November/id_1459779483796389892_2021-11-14.mp4", prompt, args)
    except Exception as e:

      if not os.path.exists(exception_path):
        df = pd.DataFrame(columns=["id", "exception", "Stacktrace"])
        df.to_csv(exception_path, index=False)

      with open(exception_path, mode="a", newline="") as exception:
        writer = csv.writer(exception)
        writer.writerow([id_string, str(e), traceback.format_exc()])


      torch.cuda.empty_cache()
      continue
  
    
    torch.cuda.empty_cache()

    print(result)

    with open(solution_path, mode="a", newline="") as solution:
      writer = csv.writer(solution)
      writer.writerow([id_string, result])

# Check if the video is in the duplicate file
# @param video_id: "id_x...x_YYYY-MM-DD"
# @param dup_set: Set with all duplicate ids created by load_csv_into_set()
# @return: Returns true if duplicate is present, returns false in every other case
def checkDuplicate(video_id : str, dup_set : set) -> bool:
    return video_id in dup_set
    
if __name__ == "__main__": 
  main()