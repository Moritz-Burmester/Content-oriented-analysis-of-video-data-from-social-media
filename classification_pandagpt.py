import torch
from PandaGPT.code.model.openllama import OpenLLAMAPEFTModel

def init_pandagpt():
    """
    Inits the model

    Return:
        Parameters of the model
    """
    
    args = {
        "model": "openllama_peft",
        "imagebind_ckpt_path": "/work/mburmest/bachelorarbeit/PandaGPT/pretrained_ckpt/imagebind_ckpt",
        "vicuna_ckpt_path": "/work/mburmest/bachelorarbeit/PandaGPT/pretrained_ckpt/vicuna_ckpt/13b_v0",
        "delta_ckpt_path": "/work/mburmest/bachelorarbeit/PandaGPT/pretrained_ckpt/pandagpt_ckpt/13b/pytorch_model.pt",
        "stage": 2,
        "max_tgt_len": 128,
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }
    model = OpenLLAMAPEFTModel(**args)
    delta_ckpt = torch.load(args["delta_ckpt_path"], map_location=torch.device("cpu"))
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.eval().half().cuda()

    max_length = 128
    top_p = 0.01
    temperature = 0.7

    return model, max_length, top_p, temperature

def classify_pandagpt(sel_video, prompts, *args):
    """
    Classifies a video for a selection of prompts
    
    Inputs:
        sel_video: Path to the video that is classified
        prompts: List of prompts used on that video
        *args: Parameters created from init() for running the model

    Return:
        List of results
    """

    model = args[0]
    max_length = args[1]
    top_p = args[2] 
    temperature = args[3]
    result = []

    for sel_prompt in prompts:
        torch.cuda.empty_cache()
        response = model.generate({
            "prompt": sel_prompt,
            "image_paths":[],
            "audio_paths":[],
            "video_paths": [sel_video],
            "thermal_paths": [],
            "top_p": top_p,
            "temperature": temperature,
            "max_tgt_len": max_length,
            "modality_embeds": []
            })
        result.append(response)
    
    return result