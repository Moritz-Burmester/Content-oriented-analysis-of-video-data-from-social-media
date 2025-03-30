import glob
import sys
import os
import traceback
import pandas as pd
import torch
import time
import re
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
PROMPTS = {
    "animals":              "Analyze the video. If the video is about animals like for example " + \
                            "pets, farm animals, polar bears, land mammals, sea mammals, fish, amphibians, reptiles, invertabrates or birds " + \
                            "answer with Yes, ... . If the video is not about animals answer with No, ... .",

    "animals_kind":         "Is the video featuring " + \
                            "pets, farm animals, polar bears, land mammals, sea mammals, fish, amphibians, reptiles, invertabrates, birds or other animals? " + \
                            "Answer as short as possible.",

    "climateactions":       "Analyze the video. If the video is about climateactions like for example " + \
                            "protests, politics, sustainable energy (wind, solar, hydropower, biogas) or fossil energy (carbon, natural gas, oil, fossil fuel) " + \
                            "answer with Yes, ... . If the video is not about climateactions answer with No, ... .",

    "climateactions_kind":  "Is the video featuring " + \
                            "protests, politics, sustainable energy like wind energy, solar energy, hydropower energy and biogas energy or fossil energy like carbon energy, natural gas energy, oil and fossil fuel? " + \
                            "Answer as short as possible.",

    "consequences":         "Analyze the video. If the video is about climate consequences like for example " + \
                            "biodeversity loss, covid, health, extrem weather (drough, flood, wildfire), melting ice, sea-level rise, rising temperatures, human rights or economic consequences " + \
                            "answer with Yes, ... . If the video is not about climate consequences answer with No, ... .",

    "consequences_kind":    "Is the video featuring " + \
                            "biodeversity loss, covid, health, extrem weather like droughs, floods and wildfire, melting ice, sea-level rise, rising temperatures, human rights or economic consequences? " + \
                            "Answer as short as possible.",

    "setting":              "There are sixteen categories with different sub-categories (...): " + \
                            "Residential area, Commercial area, Industrial area, Agriculture, Rural, Farm, Indoor Space (Room), Pole (Arctic, Antarctic, Not sure), Ocean, Coast, Desert, Forest, Jungle, Other nature, Other space. " + \
                            "Asign fitting categories to the video. If a category has a sub-category asign the fitting sub-category. If no subcategory is fitting the video write: Other. "  + \
                            "Output only the category or subcategory, without any other text like this: category1, (sub)category2,  ... . " + \
                            "If the video does not fit in any category write: Other.",

    "setting":              "Analyze the video. " + \
                            "Is the setting of the video a residential area, commercial area, industrial area, agriculture, rural, a farm, an indoor space like a room or something else, a pole like arctic or antarctic, an ocean, a coast, a desert, a forest, a jungle, other nature, other space or outer space? " + \
                            "Answer as short as possible.",

    "type":                 "Analyze the video. " + \
                            "Is the video a poster, an event invitation, a meme of climatechange, a meme in general, an infographic, a data visualization, an illustration, a text, a photo, a collage or something other? " + \
                            "Answer as short as possible.",
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

        id_string = video.split("/")[-1].split(".")[0]
        if id_string in set_processed:
            continue
        
        try:
            results = classify(video, sel_prompts, *args)
            print(results)
            
            # Format results or try again
            for idx, (response, prompt) in enumerate(zip(results, sel_prompts), 0):
                tries = 0 
                solution = response
                current_result = format_result(solution, prompt)

                while tries <= 2 and current_result == "Unknown":
                    tries += 1
                    solution = classify(video, [prompt], *args)[0]
                    current_result = format_result(solution, prompt)

                results[idx] = current_result
            
            print(results)
            # Get specific results
            for idx, response in enumerate(results):
                if response in ["animals" , "climateactions", "consequences"]:  
                    prompt_key = f"{response}_kind"
                    prompt = PROMPTS[prompt_key]
                    print("PROMTKEY: " + prompt_key)
                    tries = 0
                    solution = classify(video, [prompt], *args)[0]
                    current_result = format_result(solution, prompt)
                    print(current_result)

                    while tries <= 2 and current_result == "Unknown":
                        tries += 1
                        solution = classify(video, [prompt], *args)[0]
                        current_result = format_result(solution, PROMPTS[prompt_key])
                        print(current_result)
                    
                    if current_result != "Unknown":
                        results[idx] = current_result
            
            print(results)
            write_to_csv(SOLUTION_PATH, ["id", "animals", "climateactions", "consequences", "setting", "type"], [id_string] + results)
            print("\n\n")

        except Exception as e:

            if not os.path.exists(EXCEPTION_PATH):
                pd.DataFrame(columns=["id", "exception", "Stacktrace"]).to_csv(EXCEPTION_PATH, index=False)

            write_to_csv(EXCEPTION_PATH, ["id", "exception", "Stacktrace"], [id_string, str(e), traceback.format_exc()])
        
            
        
        # Timer for progress
        if idx % 25 == 0 or idx == total_files:
            print_progress_status(idx, total_files, start_time)

#TODO: Everything after not to the end falls out
def format_result(result: str, prompt: str):    
    result_lower = result.lower()
    prompt_categories = {
        PROMPTS["animals"]: "animals",
        PROMPTS["climateactions"]: "climateactions",
        PROMPTS["consequences"]: "consequences"
    }

    if prompt in prompt_categories:
        return prompt_categories[prompt] if "yes" in result_lower else "No" if "no" in result_lower else "Unknown"
    elif prompt == PROMPTS["animals_kind"]:
        animals_map = {
            r"pet(s|ting)?": "Pets",
            r"farm\s?animal(s)?": "Farm Animals",
            r"polar\s?bear(s)?": "Polar Bear",
            r"land\s?mammal(s)?": "Land Mammal",
            r"sea\s?mammal(s)?": "Sea Mammal",
            r"fish(es)?": "Fish",
            r"amphibian(s| creatures)?": "Amphibian",
            r"reptile(s| creatures)?": "Reptile",
            r"invertebrate(s| creatures)?": "Invertebrates",
            r"bird(s| species)?": "Birds",
            r"other\s?animal(s)?": "Other Animals"
        }
        return find_words(result, animals_map)
    
    elif prompt == PROMPTS["consequences_kind"]:
        consequences_map =  {
            r"biodiversity\s?loss": "Biodiversity Loss",
            r"covid(\-19)?": "Covid",
            r"health": "Health",
            r"extreme\s?weather": "Extreme Weather",
            r"drought(s)?": "Drought",
            r"flood(s)?": "Flood",
            r"wildfire(s)?": "Wildfire",
            r"melting\s?ice": "Melting Ice",
            r"sea[\-\s]?level\s?rise": "Sea-Level Rise",
            r"rising\s?temperature(s)?": "Rising Temperature",
            r"human\s?rights?": "Human Rights",
            r"economic\s?consequences?": "Economic Consequences",
            r"other": "Other"
        }
        return find_words(result, consequences_map)
    
    elif prompt == PROMPTS["climateactions_kind"]:
        climateactions_map = {
            r"protest(s)?": "Protest",
            r"politic(s|al)?": "Politics",
            r"sustainable\s?energy?": "Sustainable Energy",
            r"wind\s?energy?": "Wind Energy",
            r"solar\s?energy?": "Solar Energy",
            r"hydropower\s?energy?": "Hydropower Energy",
            "biogas\s?energy?": "Biogas Energy",
            r"fossil\s?energy?": "Fossil Energy",
            r"carbon\s?energy?": "Carbon Energy",
            r"natural\s?gas": "Natural Gas",
            r"oil": "Oil",
            r"fossil\s?fuel(s)?": "Fossil Fuel"
        }
        return find_words(result, climateactions_map)
    
    elif prompt == PROMPTS["setting"]:
        setting_map = {
            r"residential\s?area": "Residential Area",
            r"commercial\s?area": "Commercial Area",
            r"industrial\s?area": "Industrial Area",
            r"agriculture": "Agriculture",
            r"rural": "Rural",
            r"farm": "Farm",
            r"indoor\s?space": "Indoor Space",
            r"room": "Room",
            r"pole": "Pole",
            r"arctic": "Arctic",
            r"antarctic": "Antarctic",
            r"ocean": "Ocean",
            r"coast": "Coast",
            r"desert": "Desert",
            r"forest": "Forest",
            r"jungle": "Jungle",
            r"other\s?nature": "Other Nature",
            r"outer\s?space": "Outer Space",
            r"other\s?setting": "Other Setting"
        }
        return find_words(result, setting_map)
    
    elif prompt == PROMPTS["type"]:
        type_map = {
            r"poster": "Poster",
            r"event\s?invitation": "Event Invitation",
            r"meme\s?": "Meme",
            r"infographic": "Infographic",
            r"data\s?visuali(s|z)ation": "Data Visualization",
            r"illustration": "Illustration",
            r"text": "Text",
            r"photo": "Photo",
            r"collage": "Collage",
            r"other\s?type": "Other Type" 
        }
        return find_words(result, type_map)
    

    return "Unsafe"    

def find_words(text, words_mapping):
    matches = []
    for pattern, word in words_mapping.items():
        if re.search(rf"\b{pattern}\b", text, re.IGNORECASE):
            matches.append(word)
    return ", ".join(matches) if matches else "Unknown"


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
