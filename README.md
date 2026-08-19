# Audio Fingerprinting Project

**Note:** This is a refactored version from the April 2025 repo

This project implements an audio fingerprinting system combining the constellation mapping, spectrogram peak extraction, hashing techniques, and pre-selection process for robust audio matching. It aims to identify audio files based on their unique spectral patterns, making it useful for tasks such as audio similarity search.

# Instructions: 
1. Create a virtual environment and activate
```bash
 python3 create venv myvenv
 source myvenv/bin/activate
 ```

2. Install dependencies:

```bash 
pip install -r requirements.txt
```

3. Run the main script for end to end execution using the following command in the terminal:

```bash
python3 main.py
```

## Requirements:
1. Make sure you have Python 3.10+ installed.


## Importing files (Input Files)

The system takes .wav or .mp3 files as input for fingerprinting. Import the database recordings to the database_recordings directory.
```bash
'./data/database_recordings'
```
Import query files 
```bash
'./data/query_recordings'
```
## Project Structure


```bash
├── main.py                     # Main script for running the app
├── finger_print                # Where the fingerprint hashes are stored
    └── db_audio.pkl
├── notebooks                   # Contains plots and evaluation (F1, Precision and Recall)
    ├── evaluation.ipynb    
    └── plot.ipynb
├── scripts
    ├── __init__.py
    ├── audio_match.py          # Matching algorithm for query audio files
    ├── config.py               # Stored variables 
    ├── fingerprint_build.py    # Algorithm for building the fingerprint
├── requirements.txt            # Python dependencies for the project
├── README.md                  
└── data/                       # Folder for storing audio files 
    ├── database_recordings/    # Store database audio files
    └── query_recordings/       # Store query audio files

```
## Output files
Output files are located in output.txt 
Example:
```bash
query: classical.00003-snippet-10-10.wav database: classical.00003.wav pop.00003.wav classical.00017.wav
```
