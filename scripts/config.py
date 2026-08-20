# Database Config

SR_DB = 22050
N_FFT_DB = 1024
HOP_LENGTH_DB = N_FFT_DB // 2
WINDOW_SEC_DB = 1 # window length per second
TARGET_PEAKS_DB = 50



# Query Config

SR_Q = 44100
TARGET_SR = 22050
# WINDOW_LEN_Q = int(0.064 * 22050)
N_FFT_Q = 1024
HOP_LENGTH_Q = N_FFT_Q // 2
TARGET_PEAKS_Q = 50

MAX_DB_MATCHES = 20


# Preselect and Rank
PSELECT_RATIO = 0.02