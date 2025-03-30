import torch
from Video_LLaVA.videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from Video_LLaVA.videollava.conversation import conv_templates, SeparatorStyle
from Video_LLaVA.videollava.model.builder import load_pretrained_model
from Video_LLaVA.videollava.utils import disable_torch_init
from Video_LLaVA.videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def init_videollava():
    disable_torch_init()
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']

    return model, video_processor, tokenizer

def classify_videollava(sel_video, prompts, *args):
    video = sel_video
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    model = args[0]
    video_processor = args[1]
    tokenizer = args[2]

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    result = None

    for idx, inp in enumerate(prompts, 0):
        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        result[idx] = outputs

    return result
