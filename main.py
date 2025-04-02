import glob
import sys
import os
import traceback
import pandas as pd
import time
import re
import torch
from datetime import timedelta


#setting: Residential area, Commercial area, Industrial area, Agriculture, Rural, Farm, Indoor Space (Room), Pole (Arctic, Antarctic, Not sure), Ocean, Coast, Desert, Forest, Jungle, Other nature, Other space.
    #type: Poster, Event invitation, Meme of climatechange, Infographic, Data visualization, Illustration, Text, Photo, Collage
    #animals_kind: pets, farm animals, polar bears, land mammals, sea mammals, fish, amphibians, reptiles, invertabrates or birds
    #consequences_kind: biodeversity loss, covid, health, extrem weather (drough, flood, wildfire), melting ice, sea-level rise, rising temperature, human rights or economic consequence 
    #climateactions_kind: protests, politics, sustainable energy (wind, solar, hydropower, biogas) or fossil energy (carbon, natural gas, oil, fossil fuel)

"""
This file is for starting the models and modifying the prompts.
To start a model the corresponding conda environment must be chosen. 
The file works like this (Each step is a new start of the file).
1. The main prompt gets executed on all videos of the dataset. The result get saved in the solution file. 
2. The soltution file gets loaded. Depending one the result of the main prompt, the prompt animals, climateactions and consequences get executed. The result is saved in the same solution file.
3. Prompt for consequences and setting get executed. This is independent from privous results and from each other. The result is saved in the same solution file.

For the first prompt all duplicates get filtered. 
Each row in the solution file has the corresponding video id with the saved results.
Since every result gets immediately saved to the solution file, I have build a pick-up system that allows the code to pick up at the last saved result if e.g. the process gets interrupted.
"""

# File paths
DATASET_PATH = "/ceph/lprasse/ClimateVisions/Videos"
DUPLICATES_PATH = "/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/duplicates.csv"
ENV_NAME = os.environ.get("CONDA_DEFAULT_ENV")
SOLUTION_PATH = f"/work/mburmest/bachelorarbeit/Solution/{ENV_NAME}_solution.csv"
EXCEPTION_PATH = f"/work/mburmest/bachelorarbeit/{ENV_NAME}_exception.csv"

# Prompts
with open("Prompts/animals.txt", "r") as file:
    animals = file.read()

with open("Prompts/animals_kind.txt", "r") as file:
    animals_kind = file.read()

with open("Prompts/climateactions.txt", "r") as file:
    climateactions = file.read()

with open("Prompts/climateactions_kind.txt", "r") as file:
    climateactions_kind = file.read()

with open("Prompts/consequences.txt", "r") as file:
    consequences = file.read()

with open("Prompts/consequences_kind.txt", "r") as file:
    consequences_kind = file.read()

with open("Prompts/setting.txt", "r") as file:
    setting = file.read()

with open("Prompts/type.txt", "r") as file:
    videotype = file.read()

PROMPTS = {
    "animals":              animals,
    "animals_kind":         animals_kind,
    "climateactions":       climateactions,
    "climateactions_kind":  climateactions_kind,
    "consequences":         consequences,
    "consequences_kind":    consequences_kind,
    "setting":              setting,
    "type":                 videotype
}

def main():
    print(f"Selected environment: {ENV_NAME}\n")

    if not os.path.exists(SOLUTION_PATH):
        pd.DataFrame(columns=["id", "animals", "climateactions", "consequences", "setting", "type"]).to_csv(SOLUTION_PATH, index=False)

    # Select model based on environment
    model_params = select_model(ENV_NAME)
    
    
    print("Starting to classify")
    classify_model(*model_params)
    
    #Process and Save the dataframe
    df = pd.read_csv(SOLUTION_PATH)
    df = process_dataframe(df)
    df.to_csv(SOLUTION_PATH, index=False)

# Imports the needed functions for the model. Initializes the model and returns the classification method and init params
# ENV_NAME: the selected environment. "Automatic" variable
def select_model(ENV_NAME):
    global classify

    if ENV_NAME == "videollava":
        from classification_videollava import init_videollava, classify_videollava
        classify = classify_videollava
        return init_videollava()
    elif ENV_NAME == "pandagpt":
        from classification_pandagpt import init_pandagpt, classify_pandagpt
        classify = classify_pandagpt
        return init_pandagpt()
    elif ENV_NAME == "videochatgpt":
        from classification_videochatgpt import init_videochatgpt, classify_videochatgpt
        classify = classify_videochatgpt
        return init_videochatgpt()
    else:
        print("Error: Cannot find the selected model")
        sys.exit(1)
    return None

# Process DataFrame (sorting and dropping extra columns)
def process_dataframe(df):
    df["date"] = pd.to_datetime(df["id"].str.split("_").str[-1])
    df["numeric_id"] = df["id"].str.split("_").str[1].astype(int)
    return df.sort_values(by=["date", "numeric_id"]).drop(columns=["date", "numeric_id"])

# Classifies with the given classification method given by selectModel() and saves the result. Also it checks if a file has already been processed. This will work for all prompts
# classify: classification method of the chosen model
# *args:    args needed for the corresponding classification method of the chosen model
#TODO: Subtopic prompt & result format
def classify_model(*args):
    videos = glob.glob(f"{DATASET_PATH}/2019/01_January/*.mp4")

    set_processed = set()
    for path in [DUPLICATES_PATH, EXCEPTION_PATH, SOLUTION_PATH]:
        if os.path.exists(path):
            set_processed.update(pd.read_csv(path, usecols=["id"])["id"].dropna())

    start_time = time.time()
    total_files = len(videos)
    
    sel_prompts = [PROMPTS[key] for key in ["animals", "climateactions", "consequences", "setting", "type"]]

    for idx, video in enumerate(videos, 1):
        print("\n\n")
        id_string = video.split("/")[-1].split(".")[0]
        if id_string in set_processed:
            continue
        
        try:
            #Get all 
            torch.cuda.empty_cache()
            results = classify(video, sel_prompts, *args)
            print("Raw Result: ")
            print(results)
            
            # Format results or try again
            for idx, (response, prompt) in enumerate(zip(results, sel_prompts), 0):
                print("Prompt: " + prompt)
                print("Resonse: " + response)
                current_result = format_result(response, prompt)
                print("Formatted: " + current_result)

                # Asking for what kind it is
                if current_result in ["animals" , "climateactions", "consequences"]:
                    prompt_key = f"{current_result}_kind"
                    prompt_kind = PROMPTS[prompt_key]
                    print("Prompt_Kind:" + prompt_key)

                    torch.cuda.empty_cache()
                    response_kind = classify(video, [prompt_kind], *args)[0]
                    print("Response_kind: " + response_kind)

                    current_result = format_result(response_kind, prompt_kind)
                    print("Formatted_kind: " + current_result)

                print("Final Result: " + current_result)
                results[idx] = current_result
                 
            # Get specific results
            print("Final Results: ")
            print(results)
            write_to_csv(SOLUTION_PATH, ["id", "animals", "climateactions", "consequences", "setting", "type"], [id_string] + results)
            

        except Exception as e:

            if not os.path.exists(EXCEPTION_PATH):
                pd.DataFrame(columns=["id", "exception", "Stacktrace"]).to_csv(EXCEPTION_PATH, index=False)

            write_to_csv(EXCEPTION_PATH, ["id", "exception", "Stacktrace"], [id_string, str(e), traceback.format_exc()])
        
        # Timer for progress
        if idx % 25 == 0 or idx == total_files:
            print_progress_status(idx, total_files, start_time)


def format_result(result: str, prompt: str):    
    result_lower = result.lower().strip()
    result_lower = re.sub(r'<[^>]+>', '', result_lower)  # Remove HTML-like tags

    prompt_categories = {
        PROMPTS["animals"]: "animals",
        PROMPTS["climateactions"]: "climateactions",
        PROMPTS["consequences"]: "consequences"
    }

    # Handle yes/no/unknown prompts
    if prompt in prompt_categories:
        if re.search(r'\byes\b', result_lower):
            return prompt_categories[prompt]
        elif re.search(r'\bno\b', result_lower):
            return "No"
        return "Unknown"
    
    # Handle detailed category prompts
    elif prompt == PROMPTS["animals_kind"]:
        animals_map = {
            r"1": "Pets",   
            r"2": "Farm Animals",   
            r"3": "Polar Bears", 
            r"4": "Land Mammals",   
            r"5": "Sea Mammals", 
            r"6": "Fish",  
            r"7": "Amphibians",  
            r"8": "Reptiles",  
            r"9": "Invertebrates",   
            r"10": "Birds", 
            r"11": "Insects",
            r"12": "Other Animals"  
        }  
        return find_words(result_lower, animals_map)

    elif prompt == PROMPTS["consequences_kind"]:
        consequences_map = {
            r"1": "Floods",
            r"2": "Drought",
            r"3": "Wildfires",
            r"4": "Rising temperature",
            r"5": "Other extreme weather events",
            r"6": "Melting Ice",
            r"7": "Sea level rise",
            r"8": "Human rights",
            r"9": "Economic consequences",
            r"10": "Biodiversity loss",
            r"11": "Covid",
            r"12": "Health",
            r"13": "Other consequence"
}
        return find_words(result_lower, consequences_map)

    elif prompt == PROMPTS["climateactions_kind"]:
        climateactions_map = {
            r"1": "Politics",
            r"2": "Protests",
            r"3": "Solar energy",
            r"4": "Wind energy",
            r"5": "Hydropower",
            r"6": "Bioenergy",
            r"7": "Coal",
            r"8": "Oil",
            r"9": "Natural gas",
            r"10": "Other climate action"
        }
        return find_words(result_lower, climateactions_map)

    elif prompt == PROMPTS["setting"]:
        setting_map = {
            r"1": "No setting",
            r"2": "Residential area",
            r"3": "Industrial area", 
            r"4": "Commercial area",
            r"5": "Agricultural",
            r"6": "Rural",
            r"7": "Indoor space",
            r"8": "Arctic, Antarctica",
            r"9": "Ocean",
            r"10": "coastal",
            r"11": "Desert",
            r"12": "Forest, jungle",
            r"13": "Other Nature",
            r"14": "Outer space",
            r"15": "Other setting"
        }
        return find_words(result_lower, setting_map)

    elif prompt == PROMPTS["type"]:
        type_map = {
            r"1": "Event invitations",
            r"2": "Meme", 
            r"3": "Infographic",
            r"4": "Data visualization",
            r"5": "Illustration",
            r"6": "Screenshot",
            r"7": "Single photo",
            r"8": "Photo collage",
            r"9": "Other type"
        }
        return find_words(result_lower, type_map)
    
    return "Unknown"

def find_words(text, words_mapping):
    matches = []
    for pattern, word in words_mapping.items():
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(word)
    return " | ".join(matches) if matches else "Unknown"


# Appends a single row to a CSV file. Checks where to put the data based on the id
# file_path:    File to append to.
# columns:      Columns to append.
# data:         data the columns are appended with. data[0] = id
def write_to_csv(file_path, columns, data):
    df = pd.read_csv(file_path)
    id_value = data[0]
    if id_value in df['id'].values:
        df.loc[df['id'] == id_value, columns[1:]] = data[1:]
    else:
        new_row = pd.DataFrame([data], columns=columns)
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(file_path, index=False)


# Prints progress updates
def print_progress_status(processed_count, total_files, start_time):
    elapsed_time = time.time() - start_time
    remaining_time = (elapsed_time / processed_count) * (total_files - processed_count) if processed_count else 0
    print(f"Processed {processed_count}/{total_files} ({processed_count/total_files:.1%}) | "
          f"Elapsed: {timedelta(seconds=int(elapsed_time))} | "
          f"Remaining: {timedelta(seconds=int(remaining_time))} | "
          f"Total estimate: {timedelta(seconds=int(elapsed_time + remaining_time))}")

if __name__ == "__main__":
    main()
