import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange


class SimDADataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids
        }

        return example


import json
import os
import math
class ActivityNet(Dataset):
    def __init__(
            self,
            video_path: str = 'ActivityNet',
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 8,
            sample_frame_rate: int = 1,
    ):  
        id_path = os.path.join(video_path, 'densecap/train_ids.json')
        prompt_path = os.path.join(video_path, 'densecap/train.json')
        video_directory_path = os.path.join(video_path, 'videos')
        with open(id_path, 'r') as f:
            self.video_list = json.load(f)
        with open(prompt_path, 'r') as f:
            self.prompt_dict = json.load(f)
        self.video_directory_path = video_directory_path
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        self.video_list = [ids for ids in self.video_list if ids in self.video_list and ids in self.prompt_dict.keys()]
        #self.prompt_ids = None
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, index):
        video_id = self.video_list[index]
        
        duration = self.prompt_dict[video_id]['duration']
        prompt = self.prompt_dict[video_id]['sentences'][0]

        vr = decord.VideoReader(os.path.join(self.video_directory_path, video_id[2:]+'.mp4'), 
                                width=self.width, 
                                height=self.height,
                                num_threads=1
                                )
        
        start_sec = self.prompt_dict[video_id]['timestamps'][0][0]
        start_frame = math.ceil(start_sec * len(vr) / duration)
        sample_index = list(range(start_frame, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt
        }

        return example


class ActivityPormpt(Dataset):
    def __init__(
            self,
            video_path: str = 'ActivityNet',
            **kwargs
    ):  
        id_path = os.path.join(video_path, 'densecap/train_ids.json')
        prompt_path = os.path.join(video_path, 'densecap/train.json')
        #video_directory_path = os.path.join(video_path, 'videos')
        with open(id_path, 'r') as f:
            self.video_list = json.load(f)
        with open(prompt_path, 'r') as f:
            self.prompt_dict = json.load(f)
        '''
        self.video_directory_path = video_directory_path
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        self.prompt_ids = None
        '''
        self.video_list = [ids for ids in self.video_list if ids in self.video_list and ids in self.prompt_dict.keys()]
        
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, index):
        video_id = self.video_list[index]
        #duration = self.prompt_dict[video_id]['duration']
        prompt = self.prompt_dict[video_id]['sentences'][0]

        '''
        vr = decord.VideoReader(os.path.join(self.video_directory_path, video_id[2:]+'.mp4'), 
                                width=self.width, 
                                height=self.height)
        start_sec = self.prompt_dict[video_id]['timestamps'][0][0]
        start_frame = math.ceil(start_sec * len(vr) / duration)
        sample_index = list(range(start_frame, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        '''
        
        example = {
            #"pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt
        }

        return example
    

class MSR_VTT(Dataset):
    def __init__(
            self,
            video_path: str = 'MSR_VTT',
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 8,
            sample_frame_rate: int = 1,
    ):  
        id_path = os.path.join(video_path, 'densecap/train_ids.json')
        prompt_path = os.path.join(video_path, 'densecap/train.json')
        video_directory_path = os.path.join(video_path, 'videos')
        with open(id_path, 'r') as f:
            self.video_list = json.load(f)
        with open(prompt_path, 'r') as f:
            self.prompt_dict = json.load(f)
        self.video_directory_path = video_directory_path
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        self.video_list = [ids for ids in self.video_list if ids in self.video_list and ids in self.prompt_dict.keys()]
        #self.prompt_ids = None
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, index):
        video_id = self.video_list[index]
        
        duration = self.prompt_dict[video_id]['duration']
        prompt = self.prompt_dict[video_id]['sentences'][0]

        vr = decord.VideoReader(os.path.join(self.video_directory_path, video_id[2:]+'.mp4'), 
                                width=self.width, 
                                height=self.height)
        start_sec = self.prompt_dict[video_id]['timestamps'][0][0]
        start_frame = math.ceil(start_sec * len(vr) / duration)
        sample_index = list(range(start_frame, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt
        }

        return example