# Audio Fingerprinting Project
This project implements an audio fingerprinting system combining the constellation mapping, spectrogram peak extraction, hashing techniques, and pre-selection process for robust audio matching. It aims to identify audio files based on their unique spectral patterns, making it useful for tasks such as audio similarity search.

# Instructions: 
1. Create a virtual environment
`python3 create venv myvenv`

2. Install dependencies:

`pip install -r requirements.txt`

3. Run the main script by executing the following command in the terminal:

```bash
python main.py
```

## Requirements:
1. Make sure you have Python 3.9+ installed.


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
├── main.py                # Main script for processing audio and generating fingerprints
├── audio_match.py         # File that does audio matching and extracts query fingerprings
├── fingerprint_build.py   # File that does processing of database audio and reference fingerprints
├── requirements.txt       # Python dependencies for the project
├──plot.py                 # Plots a sample spectrogram
├──evaluation.py           # Evaluates Results
├── README.md              # Project documentation 
└── data/                  # Folder for storing audio files 
    ├── database_recordings/ # Store database audio files for fingerprinting
    └── query_recordings/     # Store query audio fingerprints

```
## Output files
Output files are located in output.txt 
Example:
```bash
query: classical.00003-snippet-10-10.wav database: classical.00003.wav pop.00003.wav classical.00017.wav
```
