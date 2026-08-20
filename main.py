import pickle
from pathlib import Path
from scripts.fingerprint_build import *
from scripts.config import *
from scripts.audio_match import *

"""
EC7006 CW2: Audio Identification

Reference paper: ROBUST FREQUENCY-BASED AUDIO FINGERPRINTING
Elsa Dupraz and Gael Richard 
Published: 2010

and 
An Industrial-Strength Audio Search Algorithm
Avery Li-Chun Wang
Published: 2003

"""


def fingerprintBuilder(database_path, fingerprint_path):

    fingerprint_path = Path(fingerprint_path)

    if not os.path.exists(fingerprint_path / "db_audio.pkl"):
         
        get_audio = AudioExtract(sr=SR_DB, n_fft=N_FFT_DB, hop_length=HOP_LENGTH_DB)
        spectrogram_dict = get_audio.extract_audio_spec(database_path)

        db_fingerprints_dict = {}

        # fname, spectrogram = spectrogram_dict
        # print(f"{fname}: {spectrogram}")

        for fname, spectrogram in spectrogram_dict.items():
            f"Processing {fname}"
            fp = FingerPrintBuild(spectrogram)
            peaks = fp.get_salient_peaks(sr=SR_DB, n_fft=N_FFT_DB, hop_length=HOP_LENGTH_DB, target_peak_s=TARGET_PEAKS_DB, window_sec=WINDOW_SEC_DB)
            song_hashes = fp.generate_hash(peaks)

            for hash_val, t_anchor in song_hashes:
                if hash_val not in db_fingerprints_dict:
                    db_fingerprints_dict[hash_val] = []

                db_fingerprints_dict[hash_val].append((fname, t_anchor))


        fp_path = fingerprint_path / "db_audio.pkl"

        with open(fp_path, "wb") as file:
            pickle.dump(db_fingerprints_dict, file)
            print(f"Saved fingerprints to {fp_path}")


def audioIdentification(query_path,fingerprint_path, output_path):

    get_audio = AudioExtract(sr=TARGET_SR, n_fft=N_FFT_Q, hop_length=HOP_LENGTH_Q)
    spectrograms = get_audio.extract_audio_spec(query_path)

    fingerprint_queries = {}

    for fname, spectrogram in spectrograms.items():
            f"Processing {fname}"
            fp = FingerPrintBuild(spectrogram)
            peaks = fp.get_salient_peaks(sr=TARGET_SR, n_fft=N_FFT_Q, hop_length=HOP_LENGTH_Q, target_peak_s=TARGET_PEAKS_Q, window_sec=1)
            song_hashes = fp.generate_hash(peaks)
    
            for hash_val, t_anchor in song_hashes:
                if hash_val not in fingerprint_queries:
                    fingerprint_queries[hash_val] = []
    
                fingerprint_queries[hash_val].append((fname, t_anchor))
    
    
    match = AudioMatch(fingerprint_path)

    # Get top 3 matches
    matched_songs = match.query_match(fingerprint_queries, max_candidates=MAX_DB_MATCHES)

    with open(output_path, "w") as f:
        for song, top_songs in matched_songs.items(): 
            line = f"query: {song}  database: " + " ".join(top_songs) + '\n'
            # print(line)
            f.write(line)

        print(f"Results saved in {output_path}")

if __name__ == "__main__":

    database_path = "./data/database_recordings"
    fingerprint_path = "./finger_print/"
    query_path = "./data/query_recordings"
    output_path = './output.txt'
    
    # Run program for automatic testing 
    fingerprintBuilder(database_path = database_path, fingerprint_path = fingerprint_path) 
    audioIdentification(query_path = query_path,fingerprint_path=fingerprint_path, output_path=output_path)

   


