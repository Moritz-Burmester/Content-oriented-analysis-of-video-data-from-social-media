import glob
import sys
import os
import traceback
import pandas as pd
import time
import re
import torch
from datetime import timedelta

"""
This file is used to classify based on the selected conda environment.
The file allows to select different output paths.
Beofre starting this file select an appropaite conda environment
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

    # Create solution file.
    if not os.path.exists(SOLUTION_PATH):
        pd.DataFrame(columns=["id", "animals", "climateactions", "consequences", "setting", "type"]).to_csv(SOLUTION_PATH, index=False)

    # Get model parameters and init the classification method
    model_params = select_model(ENV_NAME)

    # Starting to classify
    print("Starting to classify")
    classify_model(*model_params)
    
    #Process and sort the dataframe
    df = pd.read_csv(SOLUTION_PATH)
    df = process_dataframe(df)
    df.to_csv(SOLUTION_PATH, index=False)


def select_model(ENV_NAME: str):
    """
    Depending on the selected environment inits the model and classify method

    Args:
        ENV_NAME: Name of the environment

    Return:
        Parameters for the classification-method
    """
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
    elif ENV_NAME == "clip":
        from classification_clip import init_clip, classify_clip
        classify = classify_clip
        return init_clip()
    else:
        print("Error: Cannot find the selected model")
        sys.exit(1)
    return None

# Process DataFrame (sorting and dropping extra columns)
def process_dataframe(df):
    """
    Sorting a dataframe based on the video ids

    Args:
        df: Pandas Dataframe to sort.

    Return:
        Sorted pandas df
    """

    df["date"] = pd.to_datetime(df["id"].str.split("_").str[-1])
    df["numeric_id"] = df["id"].str.split("_").str[1].astype(int)
    return df.sort_values(by=["date", "numeric_id"]).drop(columns=["date", "numeric_id"])

def classify_model(*args):
    """
    Method used for using the different prompts on given videos.
    First the main prompts are used. Then the x_kind prompts based on the answer from the first prompts. 
    Everyanswer is formatted to fit in given categories.

    Args:
        *args: All input paramter needed for the classification method
    
    """

    # Selecting all videos
    videos = glob.glob(f"{DATASET_PATH}/2019/*/*.mp4")

    # Getting already processed video ids. 
    set_processed = set()
    for path in [DUPLICATES_PATH, EXCEPTION_PATH, SOLUTION_PATH]:
        if os.path.exists(path):
            set_processed.update(pd.read_csv(path, usecols=["id"])["id"].dropna())

    start_time = time.time()
    total_files = len(videos)
    
    # Prompts for the first rotation
    sel_prompts = [PROMPTS[key] for key in ["animals", "climateactions", "consequences", "setting", "type"]]
    if ENV_NAME == "clip":
        sel_prompts = [PROMPTS[key] for key in ["animals_kind", "climateactions_kind", "consequences_kind", "setting", "type"]]

    for idx, video in enumerate(videos, 1):

        # Limit
        if idx > 12001:
            break
        
        id_string = video.split("/")[-1].split(".")[0]

        # Skipping already processed videos
        if id_string in set_processed:
            continue
        
        try:
            # Clearing cuda cache and getting first results
            torch.cuda.empty_cache()
            results = classify(video, sel_prompts, *args)
            
            # Formatting every result and getting second round results for each kind of animals, climateaction and consequence
            for jdx, (response, prompt) in enumerate(zip(results, sel_prompts), 0):

                if ENV_NAME == "clip":
                    break

                current_result = format_result(response, prompt)
                #current_result = word_search(response, prompt)

                # Second round of results
                if current_result in ["animals" , "climateactions", "consequences"]:
                    prompt_key = f"{current_result}_kind"
                    prompt_kind = PROMPTS[prompt_key]

                    torch.cuda.empty_cache()
                    response_kind = classify(video, [prompt_kind], *args)[0]
   
                    current_result = format_result(response_kind, prompt_kind)
                    #current_result = word_search(response_kind, prompt_kind)
            
                results[jdx] = current_result
                 
            write_to_csv(SOLUTION_PATH, ["id", "animals", "climateactions", "consequences", "setting", "type"], [id_string] + results)
            
        except Exception as e:

            if not os.path.exists(EXCEPTION_PATH):
                pd.DataFrame(columns=["id", "exception", "Stacktrace"]).to_csv(EXCEPTION_PATH, index=False)

            write_to_csv(EXCEPTION_PATH, ["id", "exception", "Stacktrace"], [id_string, str(e), traceback.format_exc()])
        
        # Timer for progress
        if idx % 25 == 0 or idx == total_files:
            print_progress_status(idx, total_files, start_time)

def format_result(result: str, prompt: str):    
    """ 
    Based on the input it returns the categories by comparing the input to the corresponding map

    Args:
        result: A single result/answer string
        prompt: The prompt used to get the result
    
    Return:
        A string of all categories
    """

    result_lower = result.lower().strip()
    result_lower = re.sub(r"<[^>]+>", "", result_lower)  # Remove HTML-like tags

    prompt_categories = {
        PROMPTS["animals"]: "animals",
        PROMPTS["climateactions"]: "climateactions",
        PROMPTS["consequences"]: "consequences"
    }

    # Handle yes/no/unknown prompts
    if prompt in prompt_categories:
        if re.search(r"\byes\b", result_lower):
            return prompt_categories[prompt]
        elif re.search(r"\bno\b", result_lower):
            return "No"
        return "Failed Yes/No"
    
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
    """
    Method to find words and based on what words are found returning a sting of categories

    Args:
        text: String of the text to look at
        words_mapping: wordbank with search and return values

    Return:
        String of all categories divided by |
        "ALL" if all categories got selected
        "NO CLASS FOUND" if no category number got provided
    """

    matches = []
    for pattern, word in words_mapping.items():
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(word)

    if len(matches) == len(words_mapping):
        return "ALL"
    return "|".join(matches) if matches else "NO CLASS FOUND"

def word_seach(result: str, prompt: str):
    result_clean = result.lower().strip()
    result_clean = re.sub(r"<[^>]+>", "", result_clean)  # Remove HTML-like tags
    result_clean = re.sub(r"(not).*?(\.)", r"\1\2", result_clean) # Remove "not" to "."

    prompt_categories = {
        PROMPTS["animals"]: "animals",
        PROMPTS["climateactions"]: "climateactions",
        PROMPTS["consequences"]: "consequences"
    }

    # Handle yes/no/unknown prompts
    if prompt in prompt_categories:
        if re.search(r"\byes\b", result_clean):
            return prompt_categories[prompt]
        elif re.search(r"\bno\b", result_clean):
            return "No"
        return "Failed Yes/No"
    
    elif prompt == PROMPTS["animals_kind"]:
        animals = ["Pets", "Farm animals", "Polar bears", "Land mammals", "Sea mammals", "Fish", "Amphibians", "Reptiles", "Invertebrates", "Birds", "Insects", "Other"]
        wordbank = animals

    elif prompt == PROMPTS["consequences_kind"]:
        consequences = ["Floods", "Drought", "Wildfires", "Rising temperature", "Other extreme weather events", "Melting ice", "Sea level rise", "Human rights", "Economic consequences", "Biodiversity loss", "Covid", "Health", "Other consequence"]
        wordbank = consequences

    elif prompt == PROMPTS["climateactions_kind"]:
        climate_actions = ["Politics", "Protests", "Solar energy", "Wind energy", "Hydropower", "Bioenergy", "Coal", "Oil", "Natural gas", "Other climate action"]
        wordbank = climate_actions

    elif prompt == PROMPTS["setting"]:
        settings = [ "No setting", "Residential area", "Industrial area", "Commercial area", "Agricultural", "Rural", "Indoor space", "Arctic", "Antarctica", "Ocean", "Coastal", "Desert", "Forest", "Jungle", "Other nature", "Outer space", "Other setting"]
        wordbank = settings

    elif prompt == PROMPTS["type"]:
        types = ["Event invitations", "Meme", "Infographic", "Data visualization", "Illustration", "Screenshot", "Single photo", "Photo collage", "Other type"]
        wordbank = types
    
    found_words = []
    for word in wordbank:
        word_clean = word.lower()

        pattern = r'\b' + re.escape(word_clean) + r'\b' 
        if re.search(pattern, result):
            found_words.append(word)
    
    return "|".join(found_words) if found_words else "NO CLASS FOUND"

def write_to_csv(file_path, columns, data):
    """
    Appends a single row to a CSV file

    Args: 
        file_path:    File to append to.
        columns:      Columns to append.
        data:         data the columns are appended with. data[0] = id
    """
    df = pd.read_csv(file_path)
    id_value = data[0]
    if id_value in df["id"].values:
        df.loc[df["id"] == id_value, columns[1:]] = data[1:]
    else:
        new_row = pd.DataFrame([data], columns=columns)
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(file_path, index=False)

def print_progress_status(processed_count, total_files, start_time):
    """
    Prints progress updates

    Args:
        processed_count: Files processed
        total_files: Total number of files
        start_time: Time when processing started
    """
    elapsed_time = time.time() - start_time
    remaining_time = (elapsed_time / processed_count) * (total_files - processed_count) if processed_count else 0
    print(f"Processed {processed_count}/{total_files} ({processed_count/total_files:.1%}) | "
          f"Elapsed: {timedelta(seconds=int(elapsed_time))} | "
          f"Remaining: {timedelta(seconds=int(remaining_time))} | "
          f"Total estimate: {timedelta(seconds=int(elapsed_time + remaining_time))}")

if __name__ == "__main__":
    main()
