from simda.pipelines.pipeline_simda import SimDAPipeline
from simda.models.unet import UNet3DConditionModel
from simda.util import save_videos_grid
from omegaconf import OmegaConf
import torch
import argparse
from simda.data.dataset import ActivityPormpt
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import einops
from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
from tqdm.auto import tqdm
import os


def get_args_parser():
    parser = argparse.ArgumentParser('evaluation', add_help=False)
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml")
    return parser

def evaluate(
    evaluation_model_path: str,
    pretrained_model_path: str,
    my_model_path: str,
    evaluation_data,
    evaluate_size = 128,
    batch_size = 32,
    num_inference_steps = 50,
    guidance_scale = 12.5,
    use_inv_latent = False,
):  
    logger = get_logger(__name__, log_level="INFO")

    # accelerator settings
    accelerator = Accelerator(mixed_precision='fp16', log_with='tensorboard', project_dir=my_model_path)
    if accelerator.is_main_process:
        accelerator.init_trackers('evaluation')

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # inference model
    unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16)
    pipe = SimDAPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to(accelerator.device)
    pipe.set_progress_bar_config(disable=tuple)

    #evaluation model
    clip_model = CLIPModel.from_pretrained(evaluation_model_path)
    processor = CLIPProcessor.from_pretrained(evaluation_model_path)

    # parameters for video generation
    video_length = evaluation_data.video_length
    height = evaluation_data.height
    width = evaluation_data.width

    # prepare the dataset
    eval_dataset = ActivityPormpt(**evaluation_data)
    if evaluate_size:
        eval_dataset, test_dataset = torch.utils.data.random_split(eval_dataset, [evaluate_size, len(eval_dataset) - evaluate_size])
    eval_loader = DataLoader(eval_dataset, batch_size)

    # accelerate
    clip_model, eval_loader = accelerator.prepare(clip_model, eval_loader)

    #score = 0.0
    avg_score = 0.0

    # only show progress_bar on one card
    progress_bar = tqdm(range(len(eval_loader)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            prompt = batch['prompt_ids']

            if use_inv_latent:
                ddim_inv_latent = torch.load(f"{pretrained_model_path}/inv_latents/ddim_latent-50.pt").to(torch.float16)
                #this will definitely goes wrong
            else:
                ddim_inv_latent = None

            video = pipe(prompt, 
                        latents=ddim_inv_latent, # if prompt can be parallel, latent should be prallel
                        video_length=video_length, 
                        height=height, width=width, 
                        num_inference_steps=num_inference_steps, 
                        guidance_scale=guidance_scale
                        ).videos
            
            inputs = processor(text=prompt, 
                            images=einops.rearrange(video, 'b c f h w -> (b f) c h w'), 
                            return_tensors="pt", padding=True, do_rescale=False).to(accelerator.device)
            outputs = clip_model(**inputs)
            logits_per_text = outputs.logits_per_text

            for i in range(logits_per_text.size()[0]):
                for j in range(logits_per_text.size()[1]):
                    if not (i*video_length <= j < (i+1)*video_length):
                        logits_per_text[i, j] = 0
            #print(logits_per_text)
            score = torch.mean(logits_per_text) * logits_per_text.size()[0]
            avg_score += accelerator.gather_for_metrics(score).mean()
            accelerator.log({'score': avg_score}, step=step)

            progress_bar.update(1)
            torch.cuda.empty_cache()
            #print(step, avg_score)
    
    avg_score /= len(eval_loader)

    if accelerator.is_local_main_process:
        print('CLIPSIM:', avg_score/100)
    #save_videos_grid(video, f"./results/list.gif")

def main(args):
    evaluate(**OmegaConf.load(args.config))

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)