import librosa
import matplotlib.pyplot as plt
import matplotlib_inline
import os
import numpy as np
from scipy.signal import find_peaks
import json
from pathlib import Path
import hashlib
from collections import defaultdict


def extract_frame_peaks(spectrogram, sr, n_fft, hop_length):
    '''
    Extracts peak per frame and groups them into L frames per second with constellation hashing
    
    Args: 
    spectrogram(arr): processed spectrogram files
    sr(int): sample rate of 22050Hz
    n_fft(int): FFT size
    hop_length(int): hop length

    Return:
    fingerprint(list): A list of grouped peaks that represents the audio over segments
    hashes(list): List of hash objects for constellation matching
    '''
 
    frames_per_sec = sr / hop_length
    L = int(frames_per_sec * 1)  # 1-second groups
    n_peaks = 50
    frequency_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    fingerprint = []
    hashes = []
    current_group = []
    current_group_frames = 0

    for frame_index in range(spectrogram.shape[1]):
        frame = spectrogram[:, frame_index]
        time_in_seconds = frame_index * hop_length / sr
        
        # Extract and process peaks directly
        peaks_tuple = find_peaks(frame, prominence=0.1)
        peaks = peaks_tuple[0]
        prominences = peaks_tuple[1]['prominences']
        
        # Peak selection
        if len(peaks) > n_peaks:
            top_idx = np.argsort(prominences)[::-1][:n_peaks]
            peaks = peaks[top_idx]
            prominences = prominences[top_idx]
        
        # Convert to peak objects and add to current group
        frame_peaks = [{
            'frequency': float(frequency_bins[p]),
            'magnitude': float(frame[p]),
            'time': time_in_seconds,
            'frame': frame_index
        } for p in peaks]
        
        current_group.extend(frame_peaks)
        current_group_frames += 1

        # Process group when we have 1 second of frames
        if current_group_frames >= L:
            # Sort and truncate group peaks
            current_group.sort(key=lambda x: x['magnitude'], reverse=True)
            if len(current_group) > n_peaks * L:
                current_group = current_group[:n_peaks * L]
            
            # Generate hashes directly from the group peaks
            sorted_peaks = sorted(current_group, key=lambda x: x['frequency'])
            for i in range(len(sorted_peaks)):
                anchor = sorted_peaks[i]
                for j in range(i+1, min(i+10, len(sorted_peaks))):
                    target = sorted_peaks[j]
                    hash_input = f"{int(anchor['frequency'])}|{int(target['frequency'])}|{int(target['frequency'] - anchor['frequency'])}"
                    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:16]
                    
                    hashes.append({
                        'hash': hash_value,
                        'anchor_time': anchor['time'],
                        'target_time': target['time'],
                        'anchor_freq': anchor['frequency'],
                        'target_freq': target['frequency']
                    })
            
            fingerprint.append(current_group)
            current_group = []
            current_group_frames = 0

    # Convert list of hashes to dictionary format needed by rank_candidates
    hash_table = defaultdict(list)
    for hash_item in hashes:
        hash_table[hash_item['hash']].append(hash_item)

    return fingerprint, hash_table




def fingerprintBuilder(database_path='./data/database_recordings', fingerprint_path=None):
    '''
    Build grouped fingerprints and constellation hash database with immediate hash extraction and storage
    
    Args:
        database_path (str): Path to database audio files
        fingerprint_path (str): Path to save fingerprints
       
    Returns:
        None
    '''
    sample_rate = 22050
    hash_database = {}
    
    # Create directory if needed
    if fingerprint_path:
        Path(fingerprint_path).mkdir(parents=True, exist_ok=True)
    
    # Process audio files one by one
    audio_files = [os.path.join(database_path, f) for f in os.listdir(database_path) if f.endswith('.wav')]
    file_names = [os.path.basename(f) for f in audio_files]
    file_id_mapping = {str(i): name for i, name in enumerate(file_names)}
    
    # Process each file individually
    for i, audio_file in enumerate(audio_files):
        file_name = file_names[i]
        print(f"Processing {file_name}...")
        
        # Load and process audio
        y, sr = librosa.load(audio_file, sr=sample_rate)
        n_fft = int(0.064 * sr)
        hop_length = n_fft // 2
        spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann'))
        
        # Extract fingerprint and hashes
        fingerprint, hash_table = extract_frame_peaks(spectrogram, sr, n_fft, hop_length)
        
        # Add hashes to database
        for hash_value, hash_items in hash_table.items():
            if hash_value not in hash_database:
                hash_database[hash_value] = []
            
            # Add each matching hash object for this file
            for hash_obj in hash_items:
                hash_database[hash_value].append({
                    'file_id': i,
                    'file_name': file_name,
                    'anchor_time': hash_obj['anchor_time']
                })
        
        # Save individual fingerprint file with hashes
        if fingerprint_path:
            base_name = os.path.splitext(file_name)[0]
            fingerprint_file = os.path.join(fingerprint_path, f"{base_name}_fingerprint.json")
            
            try:
                # Create data structure to include hashes
                fingerprint_data = {
                    'file_name': file_name,
                    'peaks': [],  # Empty to save space
                    'groups': fingerprint,
                    'hashes': [hash_item for items in hash_table.values() for hash_item in items]  # Flatten hash items
                }
                
                # Save immediately
                with open(fingerprint_file, 'w') as f:
                    json.dump(fingerprint_data, f)
                
                print(f"Saved fingerprint and {sum(len(items) for items in hash_table.values())} hashes for {file_name}")
            except Exception as e:
                print(f"ERROR saving fingerprint for {file_name}: {str(e)}")
    
    # Save hash database and mapping at the end
    if fingerprint_path:
        hash_path = os.path.join(fingerprint_path, 'hybrid_hashes.json')
        with open(hash_path, 'w') as f:
            json.dump(hash_database, f)
        print(f"Saved hash database with {len(hash_database)} unique hashes")
        
        # Save file ID mapping
        mapping_path = os.path.join(fingerprint_path, 'file_id_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(file_id_mapping, f)
        print(f"Saved file ID mapping for {len(file_names)} files")
    
    # print(f"Generated {len(hash_database)} unique hashes from {len(file_names)} audio files")
    
    return 0
# list_fingerpints = fingerprints[0:5]
# print(f'List {list_fingerpints}')




