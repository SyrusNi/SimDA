from simda.pipelines.pipeline_simda import SimDAPipeline
from simda.models.unet import UNet3DConditionModel
from simda.util import save_videos_grid
from omegaconf import OmegaConf
import torch
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('inference', add_help=False)
    parser.add_argument("--config", type=str, default="./configs/simda.yaml")
    return parser

def main(args):
    config = OmegaConf.load(args.config)
    pretrained_model_path = config.pretrained_model_path
    my_model_path = config.output_dir
    unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
    pipe = SimDAPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")

    video_length = config.train_data.n_sample_frames
    height = config.train_data.height
    width = config.train_data.width
    num_inference_steps = config.validation_data.num_inference_steps
    guidance_scale = config.validation_data.guidance_scale

    prompt = "students play football"
    #ddim_inv_latent = torch.load(f"{my_model_path}/inv_latents/ddim_latent-50.pt").to(torch.float16)
    video = pipe(prompt, 
                latents=None, 
                video_length=video_length, 
                height=height, width=width, 
                num_inference_steps=num_inference_steps, 
                guidance_scale=guidance_scale
                ).videos

    save_videos_grid(video, f"./results/{prompt}.gif")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)