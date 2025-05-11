import glob
import sys
import os
import traceback
import pandas as pd
import time
import re
import torch
from datetime import datetime, timedelta

"""
This file is used to classify based on the selected conda environment.
The file allows to select different output paths.
Beofre starting this file select an appropaite conda environment
"""

# File paths
DATASET_PATH = "/ceph/lprasse/ClimateVisions/Videos"
DUPLICATES_PATH = "/work/mburmest/bachelorarbeit/Duplicates_and_HashValues/duplicates.csv"
ENV_NAME = os.environ.get("CONDA_DEFAULT_ENV")
SOLUTION_PATH_1 = f"/work/mburmest/bachelorarbeit/Solution/{ENV_NAME}_solution_1.csv"
SOLUTION_PATH_2 = f"/work/mburmest/bachelorarbeit/Solution/{ENV_NAME}_solution_2.csv"
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

with open("Prompts/animals_kind_shuffle.txt", "r") as file:
        animals_shuffle = file.read()

with open("Prompts/climateactions_kind_shuffle.txt", "r") as file:
    climateactions_shuffle = file.read()

with open("Prompts/consequences_kind_shuffle.txt", "r") as file:
    consequences_shuffle = file.read()

with open("Prompts/setting_shuffle.txt", "r") as file:
    setting_shuffle = file.read()

with open("Prompts/type_shuffle.txt", "r") as file:
    type_shuffle = file.read()

PROMPTS = {
    "animals":                  animals,
    "animals_kind":             animals_kind,
    "animals_shuffle":          animals_shuffle,
    "climateactions":           climateactions,
    "climateactions_kind":      climateactions_kind,
    "climateactions_shuffle":   climateactions_shuffle,
    "consequences":             consequences,
    "consequences_kind":        consequences_kind,
    "consequences_shuffle":     consequences_shuffle,
    "setting":                  setting,
    "setting_shuffle":          setting_shuffle,
    "type":                     videotype,
    "type_shuffle":             type_shuffle
}

def main():
    print(f"Selected environment: {ENV_NAME}\n")

    # Create solution file.
    if not os.path.exists(SOLUTION_PATH_1):
        pd.DataFrame(columns=["id", "animals", "climateactions", "consequences", "setting", "type"]).to_csv(SOLUTION_PATH_1, index=False)

    if ENV_NAME in ["videollava", "videochatgpt", "pandagpt"] and not os.path.exists(SOLUTION_PATH_2):
        pd.DataFrame(columns=["id", "animals", "climateactions", "consequences", "setting", "type"]).to_csv(SOLUTION_PATH_2, index=False)

    # Get model parameters and init the classification method
    model_params = select_model(ENV_NAME)

    # Starting to classify
    print("Starting to classify")
    classify_model_fixed_ids(*model_params)
    #classify_model(*model_params)
    
    #Process and sort the dataframe
    for path in [SOLUTION_PATH_1, SOLUTION_PATH_2]:
        df = pd.read_csv(path)
        df = process_dataframe(df)
        df.to_csv(path, index=False)


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

    # Selecting what videos to process
    if os.path.exists("/work/mburmest/bachelorarbeit/Solution/clip_solution.csv"):
        df = pd.read_csv("/work/mburmest/bachelorarbeit/Solution/clip_solution.csv")
        id_strings = df['id'].astype(str).tolist()
        videos = [id_to_path(s) for s in id_strings]
    else:
        videos = glob.glob(f"{DATASET_PATH}/*/*/*.mp4")

    # Getting already processed video ids. 
    set_processed = set()
    for path in [DUPLICATES_PATH, EXCEPTION_PATH, SOLUTION_PATH_1, SOLUTION_PATH_2]:
        if os.path.exists(path):
            set_processed.update(pd.read_csv(path, usecols=["id"])["id"].dropna())

    start_time = time.time()
    total_files = len(videos)
    
    # Prompts for the first rotation
    first_prompts = [PROMPTS[key] for key in ["animals", "climateactions", "consequences"]]
    if ENV_NAME == "clip":
        first_prompts = [PROMPTS[key] for key in ["animals_kind", "climateactions_kind", "consequences_kind", "setting", "type"]]

    for idx, video in enumerate(videos, 1):    
        id_string = video.split("/")[-1].split(".")[0]

        # Skipping already processed videos
        if id_string in set_processed:
            continue
        
        try:
            # Clearing cuda cache and getting first results
            results = classify(video, first_prompts, *args)

            # Creating next round of results
            second_prompts = []
            no_index = []
            failed_index = []
            for jdx, (response, prompt) in enumerate(zip(results, first_prompts), 0):

                if ENV_NAME == "clip":
                    break

                prompt_categories = {
                    PROMPTS["animals"]: "animals",
                    PROMPTS["climateactions"]: "climateactions",
                    PROMPTS["consequences"]: "consequences"
                }

                response_lower = response.lower()


                if re.search(r"\byes\b", response_lower):
                    second_prompts.append(PROMPTS[f"{prompt_categories[prompt]}_kind"])
                elif re.search(r"\bno\b", response_lower):
                    no_index.append(jdx)
                else:
                    failed_index.append(jdx)
            
            second_prompts.append(PROMPTS["setting"])
            second_prompts.append(PROMPTS["type"])

            results = classify(video, second_prompts, *args)

            results_1 = format_result(results, second_prompts, no_index, failed_index)
            results_2 = word_search(results, second_prompts, no_index, failed_index)
        
            write_to_csv(SOLUTION_PATH_1, ["id", "animals", "climateactions", "consequences", "setting", "type"], [id_string] + results_1)
            write_to_csv(SOLUTION_PATH_2, ["id", "animals", "climateactions", "consequences", "setting", "type"], [id_string] + results_2)
            
        except Exception as e:

            if not os.path.exists(EXCEPTION_PATH):
                pd.DataFrame(columns=["id", "exception", "Stacktrace"]).to_csv(EXCEPTION_PATH, index=False)

            write_to_csv(EXCEPTION_PATH, ["id", "exception", "Stacktrace"], [id_string, str(e), traceback.format_exc()])
        
        # Timer for progress
        if idx % 25 == 0 or idx == total_files:
            print_progress_status(idx, total_files, start_time)
        
        torch.cuda.empty_cache()

def classify_model_fixed_ids(*args):
    if not os.path.exists("/work/mburmest/bachelorarbeit/Solution/videollava_solution_1000.csv"):
        pd.DataFrame(columns=["id", "animals", "climateactions", "consequences", "setting", "type"]).to_csv("/work/mburmest/bachelorarbeit/Solution/videollava_solution_1000.csv", index=False)


    df = pd.read_csv("/work/mburmest/bachelorarbeit/random_ids_by_year.csv")
    id_strings = df['id'].astype(str).tolist()
    videos = [id_to_path(s) for s in id_strings]
    first_prompts = [PROMPTS[key] for key in ["animals", "climateactions", "consequences"]]

    for video in videos:    
        torch.cuda.empty_cache()
        id_string = video.split("/")[-1].split(".")[0]

        results = classify(video, first_prompts, *args)

            # Creating next round of results
        second_prompts = []
        no_index = []
        failed_index = []
        for jdx, (response, prompt) in enumerate(zip(results, first_prompts), 0):

            prompt_categories = {
                PROMPTS["animals"]: "animals",
                PROMPTS["climateactions"]: "climateactions",
                PROMPTS["consequences"]: "consequences"
            }

            response_lower = response.lower()

            if re.search(r"\byes\b", response_lower):
                second_prompts.append(PROMPTS[f"{prompt_categories[prompt]}_shuffle"])
            elif re.search(r"\bno\b", response_lower):
                no_index.append(jdx)
            else:
                failed_index.append(jdx)
            
        second_prompts.append(setting_shuffle)
        second_prompts.append(type_shuffle)
     
        results = classify(video, second_prompts, *args)
        
        results_1 = format_result_shuffle(results, second_prompts, no_index, failed_index)

        write_to_csv("/work/mburmest/bachelorarbeit/Solution/videollava_solution_1000.csv", ["id", "animals", "climateactions", "consequences", "setting", "type"], [id_string] + results_1)

def id_to_path(video_id):
    # Extract date part (after the last underscore)
    date_str = video_id.split('_')[-1]
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    year = date_obj.year
    month = date_obj.month
    month_name = date_obj.strftime('%B')  # Full month name
    
    # Create the path
    path = f"{DATASET_PATH}/{year}/{month:02d}_{month_name}/{video_id}.mp4"
    return path

def format_result(results: list, prompts: list, no_index: list, failed_index: list):    
    """ 
    Based on the input it returns the categories by comparing the input to the corresponding map

    Args:
        result: A single result/answer string
        prompt: The prompt used to get the result
    
    Return:
        A list for saving the results
    """
    solution = []
    for result, prompt in zip(results, prompts):
        
        result_lower = result.lower().strip()
        result_lower = re.sub(r"<[^>]+>", "", result_lower)  # Remove HTML-like tags
        
        # Handle detailed category prompts
        if prompt == PROMPTS["animals_kind"]:
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
            solution.append(find_words(result_lower, animals_map))

        elif prompt == PROMPTS["consequences_kind"]:
            consequences_map = {
                r"1": "Floods",
                r"2": "Drought",
                r"3": "Wildfires",
                r"4": "Rising Temperature",
                r"5": "Other Extreme Weather Events",
                r"6": "Melting Ice",
                r"7": "Sea Level Rise",
                r"8": "Human Rights",
                r"9": "Economic Consequences",
                r"10": "Biodiversity Loss",
                r"11": "Covid",
                r"12": "Health",
                r"13": "Other Consequence"
            }
            solution.append(find_words(result_lower, consequences_map))

        elif prompt == PROMPTS["climateactions_kind"]:
            climateactions_map = {
                r"1": "Politics",
                r"2": "Protests",
                r"3": "Solar Energy",
                r"4": "Wind Energy",
                r"5": "Hydropower",
                r"6": "Bioenergy",
                r"7": "Coal",
                r"8": "Oil",
                r"9": "Natural Gas",
                r"10": "Other Climate Action"
            }
            solution.append(find_words(result_lower, climateactions_map))

        elif prompt == PROMPTS["setting"]:
            setting_map = {
                r"1": "No Setting",
                r"2": "Residential Area",
                r"3": "Industrial Area", 
                r"4": "Commercial Area",
                r"5": "Agricultural",
                r"6": "Rural",
                r"7": "Indoor Space",
                r"8": "Arctic, Antarctica",
                r"9": "Ocean",
                r"10": "Coastal",
                r"11": "Desert",
                r"12": "Forest, jungle",
                r"13": "Other Nature",
                r"14": "Outer space",
                r"15": "Other setting"
            }
            solution.append(find_words(result_lower, setting_map))

        elif prompt == PROMPTS["type"]:
            type_map = {
                r"1": "Event Invitations",
                r"2": "Meme", 
                r"3": "Infographic",
                r"4": "Data Visualization",
                r"5": "Illustration",
                r"6": "Screenshot",
                r"7": "Single Photo",
                r"8": "Photo Collage",
                r"9": "Other Type"
            }
            solution.append(find_words(result_lower, type_map))

    for x in no_index:
        solution.insert(x, "No")  
    
    for x in failed_index:
        solution.insert(x, "Failed Yes/No")  

    return solution

def format_result_shuffle(results: list, prompts: list, no_index: list, failed_index: list):    
    """ 
    Based on the input it returns the categories by comparing the input to the corresponding map

    Args:
        result: A single result/answer string
        prompt: The prompt used to get the result
    
    Return:
        A list for saving the results
    """
    solution = []
    for result, prompt in zip(results, prompts):
        
        result_lower = result.lower().strip()
        result_lower = re.sub(r"<[^>]+>", "", result_lower)  # Remove HTML-like tags
        
        # Handle detailed category prompts
        if prompt == PROMPTS["animals_shuffle"]:
            animals_map = {
                r"1": "Amphibians",
                r"2": "Invertebrates",
                r"3": "Birds",
                r"4": "Fish",
                r"5": "Reptiles",
                r"6": "Pets",
                r"7": "Polar Bears",
                r"8": "Other Animals",
                r"9": "Sea Mammals",
                r"10": "Land Mammals",
                r"11": "Insects",
                r"12": "Farm Animals" 
            }  
            solution.append(find_words(result_lower, animals_map))

        elif prompt == PROMPTS["consequences_shuffle"]:
            consequences_map = {
                r"1": "Covid",
                r"2": "Other Extreme Weather Events",
                r"3": "Economic Consequences",
                r"4": "Sea Level Rise",
                r"5": "Health",
                r"6": "Human Rights",
                r"7": "Wildfires",
                r"8": "Rising Temperature",
                r"9": "Drought",
                r"10": "Other Consequence",
                r"11": "Biodiversity Loss",
                r"12": "Melting Ice",
                r"13": "Floods"
            }
            solution.append(find_words(result_lower, consequences_map))

        elif prompt == PROMPTS["climateactions_shuffle"]:
            climateactions_map = {
                r"1": "Natural Gas",
                r"2": "Wind Energy",
                r"3": "Other Climate Action",
                r"4": "Politics",
                r"5": "Coal",
                r"6": "Solar Energy",
                r"7": "Oil",
                r"8": "Hydropower",
                r"9": "Bioenergy",
                r"10": "Protests"
            }
            solution.append(find_words(result_lower, climateactions_map))

        elif prompt == PROMPTS["setting_shuffle"]:
            setting_map = {
                r"1": "Forest, Jungle",
                r"2": "Arctic, Antarctica",
                r"3": "Other Setting",
                r"4": "Indoor Space",
                r"5": "Agricultural",
                r"6": "Outer Space",
                r"7": "Ocean",
                r"8": "Residential Area",
                r"9": "Rural",
                r"10": "Other Nature",
                r"11": "No Setting",
                r"12": "Coastal",
                r"13": "Industrial Area",
                r"14": "Commercial Area",
                r"15": "Desert"
            }
            solution.append(find_words(result_lower, setting_map))

        elif prompt == PROMPTS["type_shuffle"]:
            type_map = {
                r"1": "Meme",
                r"2": "Other Type",
                r"3": "Photo Collage",
                r"4": "Illustration",
                r"5": "Data Visualization",
                r"6": "Event Invitations",
                r"7": "Screenshot",
                r"8": "Infographic",
                r"9": "Single Photo"
            }
            solution.append(find_words(result_lower, type_map))

    for x in no_index:
        solution.insert(x, "No")  
    
    for x in failed_index:
        solution.insert(x, "Failed Yes/No")  

    return solution

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

def word_search(results: list, prompts: list, no_index: list, failed_index: list):
    
    solution = []
    for result, prompt in zip(results, prompts):
        result_clean = result.lower().strip()
        result_clean = re.sub(r"<[^>]+>", "", result_clean)  # Remove HTML-like tags
        result_clean = re.sub(r"(not).*?(\.)", r"\1\2", result_clean) # Remove "not" to "."

        if prompt == PROMPTS["animals_kind"]:
            animals = ["Pets", "Farm animals", "Polar bears", "Land mammals", "Sea mammals", "Fish", "Amphibians", "Reptiles", "Invertebrates", "Birds", "Insects", "Other"]
            wordbank = animals

        elif prompt == PROMPTS["consequences_kind"]:
            consequences = ["Floods", "Drought", "Wildfires", "Rising temperature", "Other extreme weather events", "Melting ice", "Sea level rise", "Human rights", "Economic consequences", "Biodiversity loss", "Covid", "Health", "Other consequence"]
            wordbank = consequences

        elif prompt == PROMPTS["climateactions_kind"]:
            climate_actions = ["Politics", "Protests", "Solar energy", "Wind energy", "Hydropower", "Bioenergy", "Coal", "Oil", "Natural gas", "Other climate action"]
            wordbank = climate_actions

        elif prompt == PROMPTS["setting"]:
            settings = ["No setting", "Residential area", "Industrial area", "Commercial area", "Agricultural", "Rural", "Indoor space", "Arctic", "Antarctica", "Ocean", "Coastal", "Desert", "Forest", "Jungle", "Other nature", "Outer space", "Other setting"]
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
        
        solution.append("|".join(found_words) if found_words else "NO CLASS FOUND")

    for x in no_index:
        solution.insert(x, "No")  
    
    for x in failed_index:
        solution.insert(x, "FAILED YES/NO") 

    return solution

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
        new_row = pd.DataFrame(data=[data], columns=columns)
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
