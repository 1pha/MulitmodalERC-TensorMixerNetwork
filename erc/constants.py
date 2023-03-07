from collections import OrderedDict
from enum import Enum


class RunMode(Enum):
    TRAIN = "train"
    VALID = "valid"
    # Test is not used for now
    TEST = "test"


# Emotion mapper to index
emotion2idx = {
    "surprise": 1,
    "fear": 2,
    "angry": 3,
    "neutral": 4,
    "happy": 5,
    "sad": 6,
    "disgust": 7,
}

# Column names to be chosen from annotations .csv
selected_columns = OrderedDict({
    "Numb": "Numb",
    "Wav": "wav_start",
    "Unnamed: 2": "wav_end",
    "ECG": "ecg_start",
    "Unnamed: 4": "ecg_end",
    "E4-EDA": "e4-eda_start",
    "Unnamed: 6": "e4-eda_end",
    "E4-TEMP": "e4-temp_start",
    "Unnamed: 8": "e4-temp_end",
    "Segment ID": "segment_id",
    "Total Evaluation": "emotion",
    "Unnamed: 11": "valence",
    "Unnamed: 12": "arousal",
})