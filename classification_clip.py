import torch
import clip
import av
import numpy as np
import re
from PIL import Image

"""
File used for the classifcation of videos with CLIP.
"""

def init_clip():
    """
    Inits the model

    Return:
        Parameters of the model
    """

    model, preprocess = clip.load("ViT-B/32", device="cuda")
    return model, preprocess, 

def classify_clip(sel_video, prompts, *args):
    """
    Classifies a video for a selection of prompts
    
    Inputs:
        sel_video (str): Path to the video that is classified
        prompts (list of str): List of prompts used on that video
        *args: Parameters created from init() for running the model

    Return:
        List of results
    """

    model = args[0]
    preprocess = args[1]
    device = "cuda"

    frames = extract_frames(sel_video)
    solution = []
    for idx, prompt in enumerate(prompts):
        prompt_categories = format_prompt(prompt) # ["a diagram", "a dog", "a cat"]
        probabilities = []
        for frame in frames:
            image = preprocess(frame).unsqueeze(0).to(device)
            text = clip.tokenize(prompt_categories).to(device)

            with torch.no_grad():
                logits_per_image, _ = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0] # [0.9927937  0.00421068 0.00299572]
                probabilities.append(probs)
            
        solution.append(format_result(prompt_categories, probabilities, idx < 3)) # Appended with "word1" or "No" or "word1|word2"
    
    return solution

def extract_frames(video_path):
    """
    Extracts up to 11 evenly spaced frames from a video using Decord and converts them to PyTorch tensors.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        list: A list of extracted frames as PyTorch tensors.
    """

    num_frames = 11

    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames

    if total_frames < num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            img = frame.to_image()  # Convert to PIL.Image
            frames.append(img)
        if i > max(indices):
            break

    return frames

def format_prompt(prompt:str):
    """
    Extracts category labels from prompt.

    Input:
        prompt: String of a prompt

    Returns: 
        A list of strings ["a diagram", "a dog", "a cat"]
    """
    pattern = r'\d+:\s*([^:;]+)\s*;'
    categories = re.findall(pattern, prompt)
    
    # Clean and return (preserve original capitalization)
    return categories

def format_result(prompt_categories, probabilities, no_allowed):
    """
    This method contructs the result of CLIPs output by building a string of all categories that have a likelyhood of over 50%.
    If no such category can be found "No" or the highest propability category will be selected. This depends on no_allowed.

    Parameter:
        prompt_categories (list of str): ["a diagram", "a dog", "a cat"]
        probabilities (): [0.9927937  0.00421068 0.00299572]
        no_allowed (bool): True if no category can be assigned, False if a category has to be present.

    Returns:
        String. "word1" or "No" or "word1|word2"
    """

    cleaned = lambda category: re.sub(r"\s*\(.*?\)", "", category).strip()
    result = ""

    for idx, category in enumerate(prompt_categories):
        for row in probabilities:
            probability = row[idx]

            # CLIP Threshhold set at 50%
            if probability > 0.5:
                result = append_string(result, cleaned(category))
                break

    if no_allowed and result == "":
        result = "No"
    elif result == "":
        max_col = np.argmax(np.max(probabilities, axis=0))
        result = cleaned(prompt_categories[max_col])

    return result

def append_string(string:str, appendix:str):
    """
    This builds a string by dividing two or more words like this: "word1|word2"

    Parameters:
        string (str): String to append to
        appendix (str): String to append

    Returns:
        String in format: string|appendix
    """

    if string == "":
        string = appendix
    else:
        string += "|" + appendix

    return string