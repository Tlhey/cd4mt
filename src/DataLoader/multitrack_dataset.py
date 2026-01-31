# Author: David Harwath
import sys
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import utilities.audio as Audio
import os
import torchvision
import yaml
import pandas as pd
import omegaconf


class DS_10283_2325_Dataset(Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__()

        self.train = train
        self.config = config

        # Only use parameters that exist in cfg.yaml
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.mixup = config["augmentation"]["mixup"]
        self.whole_track = whole_track
        
        # Calculate segment_length from segment_duration in config
        segment_duration = config["preprocessing"]["audio"].get("segment_duration", 10.24)
        self.segment_length = int(segment_duration * self.sampling_rate)

        self.data = []
        if type(dataset_path) is str:
            self.data = self.read_datafile(dataset_path, label_path, train) 
        elif type(dataset_path) is list or type(dataset_path) is omegaconf.listconfig.ListConfig:
            for datapath in dataset_path:
                self.data +=  self.read_datafile(datapath, label_path, train) 
        else:
            raise Exception("Invalid data format")
        print("Data size: {}".format(len(self.data)))

        self.total_len = int(len(self.data) * factor)

        # Disable mixup during evaluation
        if not train:
            self.mixup = 0.0

        self.return_all_wav = False
        if self.mixup > 1:
            self.return_all_wav = config["augmentation"]["return_all_wav"] 

        print("Use mixup rate of %s" % self.mixup)

        print(f'| DS_10283_2325 Dataset Length:{len(self.data)} | Epoch Length: {self.total_len}')

    def read_datafile(self, dataset_path, label_path, train):
        file_path = dataset_path
        with open(file_path, "r") as fp:
            data_json = json.load(fp)
            data = data_json["data"]

        dataset_directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        wav_directory = os.path.abspath(os.path.join(dataset_directory, "wav_files"))

        for entry in data:
            prompt_file_path = os.path.join(wav_directory, entry["audio_prompt"])
            response_file_path = os.path.join(wav_directory, entry["audio_response"])
            entry['wav'] = prompt_file_path
            entry["response"] = response_file_path

        self.label = data
        return data

    def normalize_wav(self, x):
        x = x[0]
        x = x - x.mean()
        x = x / (torch.max(x.abs()) + 1e-8)
        x = x * 0.5
        x = x.unsqueeze(0)
        return x

    def random_segment_wav(self, x):
        wav_len = x.shape[-1]
        assert wav_len > 100, "Waveform is too short, %s" % wav_len
        if self.whole_track:
            return x
        if wav_len - self.segment_length > 0:
            if self.train:
                sta = random.randint(0, wav_len - self.segment_length)
            else:
                sta = (wav_len - self.segment_length) // 2
            x = x[:, sta: sta + self.segment_length]
        return x

    def read_wav(self, filename, frame_offset=None, num_frames=None):
        if frame_offset is not None:
            audio_data, sr = torchaudio.load(filename, frame_offset=frame_offset, num_frames=num_frames)
        else:
            audio_data, sr = torchaudio.load(filename)
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(audio_data, sr, self.sampling_rate)
        else:
            y = audio_data
        y = self.normalize_wav(y)
        y = self.random_segment_wav(y)
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y
    
    def get_mel(self, filename, mix_filename = None, frame_offset = 0):
        y = self.read_wav(filename, frame_offset)
        # Return waveform directly without mel spectrogram
        return y[0].numpy(), np.zeros((1, 1))
            
    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
        lf = self.label[idx]
        if lf is not None:
            text = lf['text']
        else:
            text = ""        
        
        prompt, fbank_prompt = self.get_mel(f["wav"], None, f["frame_offset"])
        response, fbank_response = self.get_mel(f["response"], None, f["frame_offset"])

        data_dict['fname'] = os.path.basename(lf['text']).split('.')[0]+"_from_"+str(f["frame_offset"])
        data_dict['fbank_prompt'] = fbank_prompt
        data_dict['prompt'] = prompt
        data_dict['text'] = text
        data_dict['fbank'] = fbank_response
        data_dict['waveform'] = response

        return data_dict

    def __len__(self):
        return self.total_len


class Slakh_Dataset(DS_10283_2325_Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__(dataset_path, label_path, config, train = train, factor = factor, whole_track = whole_track)  

    def read_datafile(self, dataset_path, label_path, train):
        data = []

        for entry in os.listdir(dataset_path):
            entry_path = os.path.join(dataset_path, entry)
            lp = os.path.join(entry_path, 'metadata_updated.yaml')
            if os.path.exists(lp):
                pass
            else:
                continue

            with open(lp, "r") as fp:
                label_yaml = yaml.safe_load(fp)
            data.append(label_yaml)
        
        filtered_data = []

        prompt = self.config["path"]["prompt"]
        response = self.config["path"]["response"]

        for entry in data:
            prompts = []
            responses = []

            wav_directory = os.path.join(dataset_path, entry['audio_dir'])

            for name, stem in entry['stems'].items():
                file_path = os.path.join(wav_directory, name + ".flac")
                
                if os.path.exists(file_path):
                    if stem['inst_class'] == prompt:
                        prompts.append({'path': file_path, 'duration': stem["duration"], "active_segments": stem["active_segments"]})
                    elif stem['inst_class'] == response:
                        responses.append({'path': file_path, 'duration': stem["duration"], "active_segments": stem["active_segments"]})
                else:
                    continue
               
            for prompt_entry in prompts:
                for response_entry in responses:
                    prompt_segments = set(prompt_entry['active_segments'])
                    response_segments = set(response_entry['active_segments'])
                    shared_segments = sorted(prompt_segments.intersection(response_segments))
                    
                    if shared_segments:
                        for segment in shared_segments:
                            new_entry = entry.copy()
                            new_entry['prompt'] = prompt_entry['path']
                            new_entry['response'] = response_entry['path']
                            new_entry['frame_offset'] = segment
                            filtered_data.append(new_entry)

        return filtered_data

    def read_wav(self, filename, frame_offset):
        y, sr = torchaudio.load(filename, frame_offset = frame_offset*44100, num_frames = 441000)
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)
        y = self.normalize_wav(y)
        y = self.random_segment_wav(y)
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
        
        prompt, fbank_prompt = self.get_mel(f["prompt"], None, int(f["frame_offset"]))
        response, fbank_response = self.get_mel(f["response"], None, int(f["frame_offset"]))

        data_dict['fname'] = f['audio_dir'].split('/')[0]+"_from_"+str(f["frame_offset"])
        data_dict['fbank_prompt'] = fbank_prompt
        data_dict['prompt'] = prompt
        data_dict['fbank'] = fbank_response
        data_dict['waveform'] = response

        return data_dict


class MultiSource_Slakh_Dataset(DS_10283_2325_Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__(dataset_path, label_path, config, train = train, factor = factor, whole_track = whole_track)  
        self.text_prompt = config.get('path', {}).get('text_prompt', None)

    def get_duration_sec(self, file, cache=False):
        if not os.path.exists(file):
            return 0
        try:
            with open(file + ".dur", "r") as f:
                duration = float(f.readline().strip("\n"))
        except FileNotFoundError:
            audio_info = torchaudio.info(file)
            duration = audio_info.num_frames / audio_info.sample_rate
            if cache:
                with open(file + ".dur", "w") as f:
                    f.write(str(duration) + "\n")
        return duration
    
    def filter(self, tracks, audio_files_dir):
        keep = []
        durations = []
        for track in tracks:
            track_dir = os.path.join(audio_files_dir, track)
            files = [os.path.join(track_dir, stem + ".wav") for stem in self.config["path"]["stems"]]
            
            if not files:
                continue
            
            durations_track = np.array([self.get_duration_sec(file, cache=True) * self.config['preprocessing']['audio']['sampling_rate'] for file in files])
            
            if (durations_track / self.config['preprocessing']['audio']['sampling_rate'] < 10.24).any():
                continue
            
            if (durations_track / self.config['preprocessing']['audio']['sampling_rate'] >= 640.0).any():
                print("skiping_file:", track)
                continue
            
            if not (durations_track == durations_track[0]).all():
                print(f"{track} skipped because sources are not aligned!")
                print(durations_track)
                continue
            keep.append(track)
            durations.append(durations_track[0])
        
        print(f"sr={self.config['preprocessing']['audio']['sampling_rate']}, min: {10}, max: {600}")
        print(f"Keeping {len(keep)} of {len(tracks)} tracks")

        return keep, durations, np.cumsum(np.array(durations))

    def read_datafile(self, dataset_path, label_path, train):
        data = []
        tracks = os.listdir(dataset_path)
        print(f"Found {len(tracks)} tracks.")
        keep, durations, cumsum = self.filter(tracks, dataset_path)

        for idx in range(len(keep)):
            track_info = {
                'wav_path': os.path.join(dataset_path, keep[idx]),
                'duration': durations[idx],
            }
            data.append(track_info)

        entries_to_remove = []
        max_samples = 640.0 * self.config['preprocessing']['audio']['sampling_rate']
        temp_data = []

        for entry in data:
            entry['frame_offset'] = 0
            duration = entry['duration']
            temp_data.append(entry)

            if duration > self.segment_length:
                num_copies = int((min(duration, max_samples) - self.segment_length) / self.segment_length)
                for i in range(num_copies):
                    new_entry = entry.copy()
                    new_entry['frame_offset'] = (i + 1) * self.segment_length
                    temp_data.append(new_entry)

            if duration < 0.2:
                entries_to_remove.append(entry)

        temp_data = [entry for entry in temp_data if entry not in entries_to_remove]
        data = temp_data

        return data

    def read_wav(self, filename, frame_offset):
        y, sr = torchaudio.load(filename, frame_offset=int(frame_offset*self.sampling_rate), num_frames=int(self.sampling_rate*10.24))
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def get_index_offset(self, item):
        half_interval = self.segment_length // 2
        shift = np.random.randint(-half_interval, half_interval) if self.train else 0
        offset = item["frame_offset"] + shift
        
        start, end = 0.0, item["duration"]
        
        if offset > end - self.segment_length:
            offset = max(start, offset - half_interval)
        if offset < start:
            offset = 0.0
        
        offset = offset / self.config['preprocessing']['audio']['sampling_rate']
        return item, offset

    def get_mel_from_waveform(self, waveform):
        # Return zeros as placeholder for mel spectrogram
        return np.zeros((1, 1))

    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]

        index, frame_offset = self.get_index_offset(f)

        audio_list = []
        fbank_list = []
        for stem in self.config["path"]["stems"]:
            audio, fbank = self.get_mel(os.path.join(f["wav_path"], stem+".wav"), None, frame_offset)
            audio_list.append(audio[np.newaxis, :])
            fbank_list.append(fbank[np.newaxis, :])
        
        data_dict['fname'] = f['wav_path'].split('/')[-1]+"_from_"+str(int(frame_offset))
        data_dict['fbank_stems'] = np.concatenate(fbank_list, axis=0)
        data_dict['waveform_stems'] = np.concatenate(audio_list, axis=0)
        
        data_dict['waveform'] = np.sum(data_dict['waveform_stems'], axis=0)
        data_dict['fbank'] = self.get_mel_from_waveform(data_dict['waveform'])

        if self.text_prompt is not None:
           data_dict["text"] = self.text_prompt

        return data_dict
