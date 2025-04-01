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
SOLUTION_PATH = f"/work/mburmest/bachelorarbeit/Solution/{ENV_NAME}/{ENV_NAME}_solution.csv"
EXCEPTION_PATH = f"/work/mburmest/bachelorarbeit/{ENV_NAME}_exception.csv"

# Prompts
#TODO: Only answer with what kind. -> Sort the video in the following categories. 
PROMPTS = {
    "animals":              "Analyze the video. If the video is about animals like for example " + \
                            "pets, farm animals, polar bears, land mammals, sea mammals, fish, amphibians, reptiles, invertabrates or birds " + \
                            "answer with Yes, ... . If the video is not about animals answer with No, ... .",

    "animals_kind":         "Analyze the video. What kind of animals is the video featuring? Is it "+ \
                            "pets, farm animals, polar bears, land mammals, sea mammals, fish, amphibians, reptiles, invertebrates, birds, or other animals? " + \
                            "Answer with only the relevant categories from the list, separated by commas if multiple. Do not include any extra words or explanations.",

    "climateactions":       "Analyze the video. If the video is about climateactions like for example " + \
                            "protests, politics, sustainable energy (wind, solar, hydropower, biogas) or fossil energy (carbon, natural gas, oil, fossil fuel) " + \
                            "answer with Yes, ... . If the video is not about climateactions answer with No, ... .",

    "climateactions_kind":  "Analyze the video. What kind of climate actions is the video featuring? Is it " + \
                            "protests, politics, sustainable energy (wind energy, solar energy, hydropower energy, biogas energy), or fossil energy (carbon energy, natural gas energy, oil, fossil fuel)? " + \
                            "Answer with only the relevant categories from the list, separated by commas if multiple. Do not include any extra words or explanations.",

    "consequences":         "Analyze the video. If the video is about climate consequences like for example " + \
                            "biodeversity loss, covid, health, extrem weather (drough, flood, wildfire), melting ice, sea-level rise, rising temperatures, human rights or economic consequences " + \
                            "answer with Yes, ... . If the video is not about climate consequences answer with No, ... .",

    "consequences_kind":    "Analyze the video. What kind of climate consequences is the video featuring? Is it " + \
                            "biodiversity loss, covid, health, extreme weather (droughts, floods, wildfires), melting ice, sea-level rise, rising temperatures, human rights, or economic consequences? " + \
                            "Answer with only the relevant categories from the list, separated by commas if multiple. Do not include any extra words or explanations.",

    "setting":              "Analyze the video. " + \
                            "Is the setting of the video a residential area, commercial area, industrial area, agriculture, rural, a farm, an indoor space like a room or something else, a pole like arctic or antarctic, an ocean, a coast, a desert, a forest, a jungle, other nature, other space or outer space? " + \
                            "Answer as short as possible. Only mention things that are in the video.",

    "type":                 "Analyze the video. " + \
                            "Is the video a poster, an event invitation, a meme of climatechange, a meme in general, an infographic, a data visualization, an illustration, a text, a photo, a collage or something other? " + \
                            "Answer as short as possible. Only mention things that are in the video.",
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
                    print("Formatted_kind 1: " + current_result)

                    if current_result == "Unknown": 
                        current_result = "Other"
                        print("Formatted_kind 2: " + current_result)

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
    
    # Clean the string (remove content after "not" and trailing tags/special chars)
    result_lower = re.sub(r'not.*?(?=\.|$)', '', result_lower)
    result_lower = re.sub(r'<[^>]+>', '', result_lower)  # Remove HTML-like tags
    result_lower = result_lower.strip(" .")  # Trim trailing dots/spaces

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
            r"(pet(s|ting)?)\b": "Pets",   
            r"(farm(\s?animal(s)?)?)\b": "Farm Animals",   
            r"(polar(\s?bear(s)?)?)\b": "Polar Bears", 
            r"(land(\s?mammal(s)?)?)\b": "Land Mammals",   
            r"(sea(\s?mammal(s)?)?)\b": "Sea Mammals", 
            r"(fish(es)?)\b": "Fish",  
            r"(amphibian(s| creatures)?)\b": "Amphibians",  
            r"(reptile(s| creatures)?)\b": "Reptiles",  
            r"(invertebrate(s| creatures)?)\b": "Invertebrates",   
            r"(bird(s| species)?)\b": "Birds", 
            r"(other(\s?animal(s)?)?)\b": "Other Animals"  
        }  
        return find_words(result_lower, animals_map)

    elif prompt == PROMPTS["consequences_kind"]:
        consequences_map = {
            r"(biodiversity(\s?loss)?)\b": "Biodiversity Loss",
            r"(covid(\-19)?)\b": "Covid",
            r"health\b": "Health",
            r"(drought(s)?)\b": "Droughts",
            r"(flood(s)?)\b": "Floods",
            r"(wildfire(s)?)\b": "Wildfires",
            r"(melting(\s?ice)?)\b": "Melting Ice",
            r"(sea[\-\s]?level(\s?rise)?)\b": "Sea-Level Rise",
            r"(rising(\s?temperature(s)?)?)\b": "Rising Temperature",
            r"(human(\s?rights?)?)\b": "Human Rights",
            r"(economic(\s?consequences?)?)\b": "Economic Consequences",
            r"(extreme\s?weather)\b": "Extreme Weather"
        }
        return find_words(result_lower, consequences_map)

    elif prompt == PROMPTS["climateactions_kind"]:
        climateactions_map = {
            r"(protest(s)?)\b": "Protests",
            r"(politic(s|al)?)\b": "Politics",
            r"(wind(\s?energy)?)\b": "Wind Energy",
            r"(solar(\s?energy)?)\b": "Solar Energy",
            r"(hydropower(\s?energy)?)\b": "Hydropower Energy",
            r"(biogas(\s?energy)?)\b": "Biogas Energy",
            r"(carbon(\s?energy)?)\b": "Carbon Energy",
            r"(natural(\s?gas)?)\b": "Natural Gas",
            r"oil\b": "Oil",
            r"(fossil(\s?fuel(s)?)?)\b": "Fossil Fuels",
            r"(sustainable(\s?energy)?)\b": "Sustainable Energy",
            r"(fossil(\s?energy)?)\b": "Fossil Energy"
        }
        return find_words(result_lower, climateactions_map)

    elif prompt == PROMPTS["setting"]:
        setting_map = {
            r"(industrial(\s?area)?)\b": "Industrial Area",
            r"(residential(\s?area)?)\b": "Residential Area",  # Fixed label
            r"(commercial(\s?area)?)\b": "Commercial Area",  # Fixed label
            r"agriculture\b": "Agriculture",
            r"rural\b": "Rural",
            r"farm\b": "Farm",
            r"(indoor(\s?space)?)\b": "Indoor Space",
            r"room\b": "Room",
            r"pole\b": "Pole",
            r"arctic\b": "Arctic",
            r"antarctic\b": "Antarctic",
            r"ocean\b": "Ocean",
            r"coast\b": "Coast",
            r"desert\b": "Desert",
            r"forest\b": "Forest",
            r"jungle\b": "Jungle",
            r"(other\s?nature)\b": "Other Nature",
            r"(outer\s?space)\b": "Outer Space"
        }
        return find_words(result_lower, setting_map)

    elif prompt == PROMPTS["type"]:
        type_map = {
            r"poster\b": "Poster",
            r"(event(\s?invitation)?)\b": "Event Invitation",
            r"meme\b": "Meme",
            r"infographic\b": "Infographic",
            r"(data\s?visuali(s|z)ation)\b": "Data Visualization",
            r"illustration\b": "Illustration",
            r"text\b": "Text",
            r"photo\b": "Photo",
            r"collage\b": "Collage",
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
