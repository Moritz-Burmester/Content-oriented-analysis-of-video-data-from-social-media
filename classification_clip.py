import torch
import clip
import av
import numpy as np
import re
from PIL import Image

def init_clip():
    model, preprocess = clip.load("ViT-B/32", device="cuda")
    return model, preprocess, 

def classify_clip(sel_video, prompts, *args):
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
    Extracts category labels from prompt

    Input:
        prompt: String of a prompt

    Returns: 
        A list of strings ["a diagram", "a dog", "a cat"]
    """
    pattern = r'\d+:\s*([^:;]+)\s*;'
    categories = re.findall(pattern, prompt)
    
    # Clean and return (preserve original capitalization)
    return categories

#TODO: No duplicates in result

#"word1" or "No" or "word1|word2"
# Categories for the list in a prompt
def format_result(prompt_categories, probabilities, no_allowed):
    cleaned = lambda category: re.sub(r"\s*\(.*?\)", "", category).strip()
    result = ""

    for idx, category in enumerate(prompt_categories):
        for row in probabilities:
            probability = row[idx]
            if probability > 0.5:
                result = append_string(result, cleaned(category))
                break

    if no_allowed and result == "":
        result = "No"
    elif result == "":
        max_col = np.argmax(np.max(probabilities, axis=0))
        result = cleaned(prompt_categories[max_col])

    return result

def append_string(string:str, apendix:str):
    if string == "":
        string = apendix
    else:
        string += "|" + apendix

    return string