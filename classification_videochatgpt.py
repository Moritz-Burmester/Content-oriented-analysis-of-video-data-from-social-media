import torch
from Video_ChatGPT.video_chatgpt.demo.chat import Chat
from Video_ChatGPT.video_chatgpt.eval.model_utils import initialize_model
from Video_ChatGPT.video_chatgpt.constants import *
from Video_ChatGPT.video_chatgpt.video_conversation import (default_conversation)

"""
Needs approximately 18GB of GPU memory
"""

def init_videochatgpt():
    model_name = "/work/mburmest/bachelorarbeit/Video_ChatGPT/LLaVA-7B-Lightning-v1-1"
    projection_path = "/work/mburmest/bachelorarbeit/Video_ChatGPT/video_chatgpt-7B.bin"
    temperature =  0.1
    max_output_tokens = 128

    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(model_name, projection_path)
    return model, model_name, vision_tower, tokenizer, image_processor, video_token_len, temperature, max_output_tokens

def classify_videochatgpt(sel_video, prompts, *args):
    model = args[0]
    model_name = args[1]
    vision_tower = args[2] 
    tokenizer = args[3] 
    image_processor = args[4]
    video_token_len = args[5]
    temperature = args[6]
    max_output_tokens = args[7]

    replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN

    chat = Chat(model_name, "video-chatgpt_v1", tokenizer, image_processor, vision_tower, model, replace_token)

    img_list = []
    chat.upload_video(sel_video, img_list)

    result = None

    for idx, sel_prompt in enumerate(prompts, 0):
        state = default_conversation.copy()

        if "<video>" not in sel_prompt:
            sel_prompt = sel_prompt + "\n<video>"

        state.append_message(state.roles[0], (sel_prompt, sel_video))
        state.append_message(state.roles[1], None)
        _, response = list(chat.answer(state, img_list, temperature, max_output_tokens, first_run=True))[-1][1][-1]

        result[idx] = response

    return result
