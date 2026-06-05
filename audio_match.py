import librosa
import os
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks


def audio_processing_query(sample_rate, target_sr, audio_path):

    '''
    Process all query audio files in the directory and return all spectrograms

    Args:

    sample_rate(int):  passes original sample rate of the query audio file
    target_sr(int): passes the target sample rate to match the database audio file
    audio_path(str): passes the directory of the audio path


    Return:
    all_stfts(list):
    file_names(list) :
    n_fft(int):
    hop_length(int):


    '''
    
    audio_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith('.wav')]
    n_fft =int(0.064 * target_sr)# convert 64 ms to seconds and multiply with sample rate
    hop_length = n_fft // 2 # Hop length of 50 %
    
    all_stfts = []
    file_names = []

    # Processes each audio file into a spectrograms
    for audio_file in audio_files:
        file_name = os.path.basename(audio_file)
       
        y, orig_sr = librosa.load(audio_file, sr=sample_rate) # get original sample rate from the query audio file
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr) # resample to 22050
        spectrograms = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann'))

        all_stfts.append(spectrograms)
        file_names.append(file_name)
       
    # print(f" List of audio files {list(audio_files)}")

    return all_stfts, file_names,n_fft, hop_length



def extract_query_peaks(spectrogram, sr, n_fft, hop_length):
    '''
    Extracts peak per frame and groups them into L frames per second

    Args: 
    spectrogram(arr): processed spectrogram files
    sr(int): sample rate of 22050Hz
    n_fft(int): FFT size

    Return:
    fingerprint(list): A list of grouped peaks that represents the audio over segments
    hash_table(dict): Dictionary of hashed objects, with the hash value as the key
    '''
    frames_per_sec = sr / hop_length  # frames per second
    L = 30  # number of frames per second
    n_peaks = 100  # number of peaks retained per frame 
    
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


def preselect_hash_candidates(query_hashes, hash_database):
    '''
    Preselect candidate files based on hash match count ranking
   
    Args:
        query_hashes (dict): Hash table from extract_query_peaks function
        hash_database (dict): Full hash database mapping hash values to file matches
        
       
    Returns:
        filtered_hash_db(dict): Filtered hash database containing only preselected candidates
    '''

    
    # Count how many hashes match each file
    file_hash_counts = {}
   
    # Extract hash keys from query_hashes
    query_hash_values = set(query_hashes.keys())
    total_query_hashes = len(query_hash_values)
   
    # print(f"Query has {total_query_hashes} unique hashes")
   
    # Build file-to-hash-count mapping
    for hash_value in query_hash_values:
        if hash_value in hash_database:
            for match in hash_database[hash_value]:
                file_name = match['file_name']
                if file_name not in file_hash_counts:
                    file_hash_counts[file_name] = 0
                file_hash_counts[file_name] += 1
   
    # No candidates found
    if not file_hash_counts:
        print("No candidates found with matching hashes.")
        return {}
   
    # Rank candidates by hash count
    sorted_candidates = sorted(file_hash_counts.items(), key=lambda x: x[1], reverse=True)
    max_candidates = int(len(sorted_candidates) * 0.80)
    
    # Select only the top N candidates
    top_candidates = sorted_candidates[:max_candidates]
    
    # print(f"Top {len(top_candidates)} candidates from {len(file_hash_counts)} total:")
    for i, (file_name, count) in enumerate(top_candidates):
        ratio = count / total_query_hashes
        # print(f"  {i+1}. {file_name}: {count} matches ({ratio:.2%})")
    
    # Create a set of selected files
    preselected_files = {file_name for file_name, _ in top_candidates}
    
    # Create a filtered hash database with only the preselected files
    filtered_hash_db = {}
    for hash_value in hash_database:
        filtered_matches = [match for match in hash_database[hash_value]
                          if match['file_name'] in preselected_files]
        if filtered_matches:
            filtered_hash_db[hash_value] = filtered_matches
   
    print(f"Preselection: Reduced from {len(file_hash_counts)} to {len(preselected_files)} candidates " +
          f"({len(preselected_files)/len(file_hash_counts)*100:.1f}%)")
   
    return filtered_hash_db


def load_hash_database(fingerprint_path):
    '''
    Load constellation hash database from file
    
    Args:
        fingerprint_path (str): Path to fingerprint directory
        
    Returns:
        hash_database (dict): Hash database
        file_id_to_path (dict): Mapping from file IDs to file paths
    '''
    hash_path = os.path.join(fingerprint_path, 'hybrid_hashes.json')
    mapping_path = os.path.join(fingerprint_path, 'file_id_mapping.json')
    
    hash_database = {}
    file_id_to_path = {}
    
    # Load hash database
    if os.path.exists(hash_path):
        with open(hash_path, 'r') as f:
            serialized_db = json.load(f)
            
            # Convert back to proper format
            for hash_value, matches in serialized_db.items():
                hash_database[hash_value] = matches
        print(f"Loaded hash database with {len(hash_database)} unique hashes")
    else:
        print(f"Hash database not found at {hash_path}")
    
    # Load file ID mapping
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            file_id_to_path = json.load(f)
        print(f"Loaded file ID mapping with {len(file_id_to_path)} entries")
    else:
        print(f"File ID mapping not found at {mapping_path}")
    
    return hash_database, file_id_to_path


def rank_candidates(query_hashes, hash_database, top_n=10):
    '''
    Simple candidate ranking for audio identification
    
    Args:
        query_hashes (dict): Hash table from query audio
        hash_database (dict): Full hash database
        top_n (int): Number of top candidates to return
        
    Returns:
        ranked_candidates (list): Ranked list of candidate files
        filtered_hash_db (dict): Filtered hash database with only ranked candidates
    '''
    # Count matching hashes per file
    file_hash_counts = {}
    # Track time offsets
    time_offsets = {}
    
    query_hash_values = set(query_hashes.keys())
    total_query_hashes = len(query_hash_values)
    
    if total_query_hashes == 0:
        return [], {}
    
    print(f"Query has {total_query_hashes} unique hashes")
    
    # Build file-to-hash-count mapping and collect time offsets
    for hash_value in query_hash_values:
        if hash_value in hash_database:
            for db_match in hash_database[hash_value]:
                file_name = db_match['file_name']
                
                # Initialise structures for this file if needed
                if file_name not in file_hash_counts:
                    file_hash_counts[file_name] = 0
                    time_offsets[file_name] = defaultdict(int)
                
                # For each query match with this hash
                for query_match in query_hashes[hash_value]:
                    query_time = query_match['anchor_time']
                    db_time = db_match['anchor_time']
                    
                    # Calculate time offset (query time - database time)
                    offset = round(query_time - db_time, 1)  # Round to nearest 0.1s
                    time_offsets[file_name][offset] += 1
                    
                    # Increment hash count
                    file_hash_counts[file_name] += 1
    
    # No matches found
    if not file_hash_counts:
        return [], {}
    
    # Calculate scores and create candidate list
    candidates = []
    for file_name, hash_count in file_hash_counts.items():
        # Find best time offset (most consistent alignment)
        best_offset, offset_count = max(time_offsets[file_name].items(), key=lambda x: x[1], default=(0, 0))
        
        # Calculate score based on hash count and alignment consistency
        score = hash_count * (offset_count / hash_count if hash_count > 0 else 0)
        
        candidates.append((file_name, {
            'score': score,
            'hash_count': hash_count,
            'best_offset': best_offset,
            'best_offset_count': offset_count
        }))
    
    # Sort by score (descending)
    sorted_candidates = sorted(candidates, key=lambda x: x[1]['score'], reverse=True)
    
    # Take top N candidates
    top_candidates = sorted_candidates[:min(top_n, len(sorted_candidates))]
    
    # Print ranking info
    print(f"Top {len(top_candidates)} candidates from {len(file_hash_counts)} total:")
    for i, (file_name, metrics) in enumerate(top_candidates):
        print(f"  {i+1}. {file_name}: score={metrics['score']:.1f}, matches={metrics['hash_count']}")
    
    # Create filtered hash database
    preselected_files = {file_name for file_name, _ in top_candidates}
    filtered_hash_db = {}
    
    for hash_value in hash_database:
        filtered_matches = [match for match in hash_database[hash_value]
                          if match['file_name'] in preselected_files]
        if filtered_matches:
            filtered_hash_db[hash_value] = filtered_matches
    
    # print(f"Ranking: Reduced from {len(file_hash_counts)} to {len(preselected_files)} candidates")
    return top_candidates, filtered_hash_db



def match_constellation_hashes(query_hash_table, hash_database, file_id_to_path=None):
    '''
    Match query hashes against hash database
    
    Args:
        query_hash_table (dict/defaultdict): Hash table from extract_query_peaks
        hash_database (dict): Hash database mapping hash values to file matches
        file_id_to_path (dict): Optional mapping from file IDs to file paths
        
    Returns:
        matches (list): List of matches sorted by score (highest first)
    '''
    # Count matches for each file and time offset
    match_counts = defaultdict(lambda: defaultdict(int))
    
    # For each query hash
    for hash_key, query_matches in query_hash_table.items():
        # Check if this hash exists in the database
        if hash_key in hash_database:
            # For each anchor point in the query with this hash
            for query_match in query_matches:
                query_time = query_match['anchor_time']
                
                # Match with all database entries with the same hash
                for db_match in hash_database[hash_key]:
                    file_name = db_match['file_name']
                    db_time = db_match['anchor_time']
                    
                    # Calculate time offset (query time - database time)
                    time_offset = query_time - db_time
                    
                    # Round the offset to handle slight variations
                    binned_offset = round(time_offset, 1)  # Round to nearest 0.1 seconds
                    
                    # Increment the count for this file and offset
                    match_counts[file_name][binned_offset] += 1
    
    # Find the best time offset for each file
    matches = []
    for file_name, offsets in match_counts.items():
        if not offsets:
            continue
            
        # Find offset with the most matches
        best_offset, match_count = max(offsets.items(), key=lambda x: x[1])
        
        # Count total matches across all offsets for this file
        total_matches = sum(offsets.values())
        
        matches.append({
            'file_name': file_name,
            'score': match_count,
            'confidence': match_count / len(query_hash_table) if query_hash_table else 0,
            'time_offset': best_offset,
            'match_consistent': match_count,  # For compatibility with traditional approach
            'total_freq_matches': total_matches,  # Total matches across all time offsets
            'hash_based': True  # Flag to indicate this is from hash-based matching
        })
    
    # Sort by score (descending)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches

def audioIdentification(query_path, fingerprint_path, output_path):
    '''
    Main function to match query audio files using hash-based matching with advanced candidate ranking
    
    Args:
        query_path (str): Path to directory with query audio files
        fingerprint_path (str): Path to directory with fingerprint database
        output_path (str): Path to save identification results
       
    Returns:
        list: List of match results including top-3 matches per query
    '''
    # Load only the hash database
    try:
        hash_database, file_id_to_path = load_hash_database(fingerprint_path)
        if len(hash_database) == 0:
            raise ValueError("Hash database is empty.")
        print(f"Loaded hash database with {len(hash_database)} unique hashes")
    except Exception as e:
        print(f"Error loading hash database: {e}")
        return []
    
    sample_rate = 44100  # sample rate of query
    target_sr = 22050  # target sample rate
    spectrograms, file_names, n_fft, hop_length = audio_processing_query(sample_rate, target_sr, query_path)
    
    use_candidate_ranking = True  # Flag to enable/disable advanced ranking
    all_query_matches = []
    
    with open(output_path, 'w') as out_file:
        for i, spectrogram in enumerate(spectrograms):
            query_name = file_names[i]
            
            # Extract query hashes
            start_time = time.time()
            _, query_hashes = extract_query_peaks(spectrogram, target_sr, n_fft, hop_length)
            extraction_time = time.time() - start_time
            # print(f"Extracted {len(query_hashes)} hashes for {query_name} in {extraction_time:.3f} seconds")
            
            # Apply advanced candidate ranking
            working_hash_db = hash_database
            if use_candidate_ranking:
                start_time = time.time()
                ranked_candidates, working_hash_db = rank_candidates(query_hashes, hash_database)
                ranking_time = time.time() - start_time
                # print(f"Candidate ranking took {ranking_time:.3f} seconds")
                
                # Get directly ranked matches if available
                if ranked_candidates:
                    # Convert ranked candidates to match format
                    matches = []
                    for file_name, metrics in ranked_candidates:
                        # Calculate time coherence if not present
                        time_coherence = (metrics['best_offset_count'] / metrics['hash_count'] 
                                        if metrics['hash_count'] > 0 else 0)
                        
                        matches.append({
                            'file_name': file_name,
                            'score': metrics['score'],  
                            'confidence': time_coherence, 
                            'time_offset': metrics['best_offset'],
                            'match_consistent': metrics['best_offset_count'],
                            'total_freq_matches': metrics['hash_count'],
                            'hash_based': True
                        })
                else:
                    # Fallback to traditional matching if no candidates
                    matches = match_constellation_hashes(query_hashes, working_hash_db)
                # Traditional matching
                start_time = time.time()
                matches = match_constellation_hashes(query_hashes, working_hash_db)
                hash_matching_time = time.time() - start_time
                # print(f"Hash matching took {hash_matching_time:.3f} seconds, found {len(matches)} matches")
            
            # Get top 3 matches for this query
            top_matches = matches[:min(3, len(matches))]
            
            # Add to query matches list
            all_query_matches.append({
                'query_file': query_name,
                'matches': [
                    {
                        'file_name': match['file_name'],
                        'score': match['score'],
                        'confidence': match.get('confidence', 0),
                        'time_offset': match.get('time_offset', 0)
                    }
                    for match in top_matches
                ]
            })
            
            # Display on console
            print(f"\nquery: {query_name} database: ", end="")

            if len(matches) == 0:
                print("No matches found")
            else:
                match_strings = []
                for j in range(min(3, len(matches))):
                    match_strings.append(matches[j]['file_name'])
                print(" ".join(match_strings))
            
            # File output.txt
            out_file.write(f"query: {query_name} database: ")
            
            if len(matches) == 0:
                out_file.write("No matches found\n")
            else:
                match_strings = []
                for j in range(min(3, len(matches))):
                    match_strings.append(matches[j]['file_name'])
                out_file.write(" ".join(match_strings) + "\n" )
    
    print(f"\nResults saved to {output_path}")
    return all_query_matches