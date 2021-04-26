import os
import json
import pandas as pd
import string
import random
from audio_utils import audioread, audiowrite
from tqdm import tqdm
from shutil import copyfile
from noise_utils import apply_noise
from text_utils import normalizeString


class Config:
    def __init__(self, config_path = 'config.json'):
        with open(config_path,'r') as f:
            cfg = json.load(f)
            
        self.snr = cfg['snr']
        self.noise_types = cfg['noise_types']
        self.transcriptions_path = cfg['transcriptions']
        self.noise_dir = cfg['noise_dir']
        self.clean_speech_dir = cfg['clean_speech_dir']
        self.results_dir = cfg['results_dir']
        self.noise_paths = []
        self.clean_paths = []
        self.config_path = config_path
        self.transcriptions = self.prepareTranscriptions()
        self.preparePaths()
        
    


    def prepareTranscriptions(self):
        df = pd.read_csv(self.transcriptions_path, sep="\t")
        transcriptions = {}
        
        for file in os.listdir(self.clean_speech_dir):
            file_name = os.path.splitext(file)[0]
            item = df[df['path'] == file_name]

            if(len(item)==1):
                sentence = item['sentence'].ravel()[0]
                transcriptions[os.path.join(self.clean_speech_dir,file)] = normalizeString(sentence)
        
        return transcriptions
    
    
    def preparePaths(self):
        for file in os.listdir(self.clean_speech_dir):
            self.clean_paths.append(os.path.join(self.clean_speech_dir,file))
            
        for _type in os.listdir(self.noise_dir):
            if _type in self.noise_types:
                for file in os.listdir(os.path.join(self.noise_dir, _type)):
                    self.noise_paths.append(os.path.join(self.noise_dir, _type, file))
        
        random.shuffle(self.noise_paths)
    
    
    def getRandomNoiseFile(self):
        return self.noise_paths[random.randint(0,len(self.noise_paths)-1)]
        