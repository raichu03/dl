import os
import pandas as pd

def verify_processing(metadata: str):
    """Verify that processing was successful"""
    metadata = pd.read_csv(metadata)
    missing_files = 0
    
    for idx, row in metadata.iterrows():
        fold = row['fold']
        filename = row['slice_file_name']
        mfcc_path = f'processed_data/mfccs/fold{fold}/{filename[:-4]}.pkl'
        
        if not os.path.exists(mfcc_path):
            missing_files += 1
            
    print(f"Total files: {len(metadata)}")
    print(f"Missing processed files: {missing_files}")
    print(f"Success rate: {(len(metadata)-missing_files)/len(metadata)*100:.2f}%")

def create_mfcc_metadata(metadata: str):
    """
    Create a metadata file for the MFCC data
    """
    original_metadata = pd.read_csv(metadata)
    processed_metadata = []
    
    for idx, row in original_metadata.iterrows():
        fold = row['fold']
        filename = row['slice_file_name']
        base_name = filename[:-4]  # Remove .wav
        
        mfcc_path = f'processed_data/mfccs/fold{fold}/{base_name}.pkl'
        
        if os.path.exists(mfcc_path):
            processed_metadata.append({
                'original_file': filename,
                'fold': fold,
                'class': row['class'],
                'classID': row['classID'],
                'mfcc_path': mfcc_path
            })
    
    processed_df = pd.DataFrame(processed_metadata)
    processed_df.to_csv('processed_data/mfcc_metadata.csv', index=False)
    print(f"Saved metadata for {len(processed_df)} MFCC files")


if __name__ == "__main__":
    
    metadata_path = 'UrbanSound8K.csv'
    verify_processing(metadata=metadata_path)
    create_mfcc_metadata(metadata=metadata_path)