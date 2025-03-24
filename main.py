import glob
import sys
import os
import traceback
import pandas as pd
import torch
import time
from datetime import timedelta

"""
This file is for starting and using the models. First select a conda environment and update the paths accordingly.
Then choose a prompt. 
Each model will analyze each file indepent to each other. 
After this the file can be used and depending on your chosen environment
"""

# File paths
dataset_path = "/ceph/lprasse/ClimateVisions/Videos"
duplicates = "/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/duplicates.csv"
env_name = os.environ.get("CONDA_DEFAULT_ENV")
solution_path = os.path.join("/work/mburmest/bachelorarbeit/", env_name + "_solution.csv")
exception_path = os.path.join("/work/mburmest/bachelorarbeit/", env_name + "_exception.csv")

# Prompt
prompt = "What is the main color in the video?"

def main():
    print(f"Selected: {env_name}" + "\n")

    # Select model based on environment
    model, *model_params = select_model(env_name)
    if not model:
        print("Error: Cannot find the selected model")
        sys.exit(1)

    classify_model(model, *model_params)

    # Sort the dataframe
    if os.path.exists(solution_path):
        df = pd.read_csv(solution_path)
        df["date"] = pd.to_datetime(df["id"].str.split("_").str[-1])
        df["numeric_id"] = df["id"].str.split("_").str[1].astype(int)
        df = df.sort_values(by=["date", "numeric_id"])
        df.drop(columns=["date", "numeric_id"]).to_csv(solution_path, index=False)

"""Imports the needed functions for the model. Initializes the model and returns the classification method and init params"""
def select_model(env_name):
    if env_name == "videollava":
        from classification_videollava import init_videollava, classify_videollava
        return (classify_videollava, *init_videollava())
    elif env_name == "pandagpt":
        from classification_pandagpt import init_pandagpt, classify_pandagpt
        return (classify_pandagpt, *init_pandagpt())
    elif env_name == "videochatgpt":
        from classification_videochatgpt import init_videochatgpt, classify_videochatgpt
        return (classify_videochatgpt, *init_videochatgpt())
    return None

"""Classifies with the given classification method given by selectModel() and saves the result. Also it checks if a file has already been processed"""
def classify_model(classify, *args):
    
    set_processed = set()
    if os.path.exists(duplicates):
        set_processed.update(load_csv_into_set(duplicates, "id"))
    if os.path.exists(solution_path):
        set_processed.update(load_csv_into_set(solution_path, "id"))
    if os.path.exists(exception_path):
        set_processed.update(load_csv_into_set(exception_path, "id"))

    videos = glob.glob(f"{dataset_path}/2019/01_January/*.mp4")

    start_time = time.time()
    processed_count = 0
    total_files = len(videos)

    for video in videos:
        id_string = video.split("/")[-1].split(".")[0]
        processed_count += 1

        if id_string in set_processed:
            continue

        try:
            result = classify(video, prompt, *args)
            write_to_csv(solution_path, ["id", "solution"], [id_string, result])
        except Exception as e:
            write_to_csv(solution_path, ["id", "solution"], [id_string, ""])
            write_to_csv(exception_path, ["id", "exception", "Stacktrace"], [id_string, str(e), traceback.format_exc()])
            torch.cuda.empty_cache()
            continue

        torch.cuda.empty_cache()
    
        if processed_count % 25 == 0 or processed_count == total_files:
            print_progress_status(processed_count, total_files, start_time)

"""Appends a single row to a CSV file."""
def write_to_csv(file_path, columns, data):    
    file_exists = os.path.exists(file_path)
    df = pd.DataFrame([data], columns=columns)
    df.to_csv(file_path, mode="a", index=False, header=not file_exists)


"""Loads a specified column from a CSV into a set for fast lookups."""
def load_csv_into_set(file_path: str, col_name: str) -> set:
    try:
        return set(pd.read_csv(file_path, usecols=[col_name], dtype=str)[col_name].dropna())
    except Exception as e:
        print(f"Error in load_csv_into_set(): {e}")
        sys.exit(1)

"""To measure the progress and to give rough estimates about the needed time"""
def print_progress_status(processed_count, total_files, start_time):
    elapsed_time = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
    
    if processed_count > 0:
        time_per_file = elapsed_time / processed_count
        remaining_files = total_files - processed_count
        remaining_time = time_per_file * remaining_files
        remaining_str = str(timedelta(seconds=int(remaining_time)))
        total_estimate_str = str(timedelta(seconds=int(elapsed_time + remaining_time)))
        
        print(f"Processed {processed_count}/{total_files} files "
              f"({processed_count/total_files:.1%}) | "
              f"Elapsed: {elapsed_str} | "
              f"Remaining: {remaining_str} | "
              f"Total estimate: {total_estimate_str}")

if __name__ == "__main__": 
    main()
