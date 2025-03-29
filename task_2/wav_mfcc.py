import os

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

def load_audio(file_path: str, target_length: str = 5,  sr: int = 22050):
    """
    Load audio file and maintain the fixed length and returns the audio and sample rate
    """
    try:
        audio, sr = librosa.load(file_path, sr=sr)
    
        ## Padding or Trimming to target length
        target_samples = target_length * sr
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
            
        return audio, sr
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def generate_mfcc(audio, sr=22050, n_mfcc=40):
    """
    Create MFCC features and return
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = librosa.util.normalize(mfcc)
    return mfcc

def create_dirs():
    """
    Create directoies to store MFCC data
    """
    os.makedirs('processed_data/mfccs', exist_ok=True)
    for fold in range(1, 11):
        os.makedirs(f'processed_data/mfccs/fold{fold}', exist_ok=True)

def main(metadata_path: str):
    """Process all audio files and save features"""
    metadata = pd.read_csv(metadata_path)
    create_dirs()
    
     # Process each file
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        fold = row['fold']
        filename = row['slice_file_name']
        class_id = row['classID']
        
        # Paths for saving
        mfcc_path = f'processed_data/mfccs/fold{fold}/{filename[:-4]}.pkl'
        # Skip if already processed
        if os.path.exists(mfcc_path):
            continue
            
        # Load and process audio
        audio_path = f'urban_data/fold{fold}/{filename}'
        audio, sr = load_audio(audio_path)
        if audio is None:
            continue
        
        # Extract features
        mfcc = generate_mfcc(audio)
        
        # Save MFCC with metadata
        data_to_save = {
            'features': mfcc,
            'class_id': class_id,
            'original_file': filename,
            'sample_rate': sr,
            'n_mfcc': 40
        }
        
        with open(mfcc_path, 'wb') as f:
            pickle.dump(data_to_save, f)
    

if __name__ == "__main__":
    metadata_path = "UrbanSound8K.csv"
    main(metadata_path=metadata_path)
    print("\nTask Completed")