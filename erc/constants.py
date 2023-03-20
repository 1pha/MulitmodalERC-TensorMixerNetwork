from collections import OrderedDict
from enum import Enum


class Task(Enum):
    CLS = "classification"
    REG = "regression"
    ALL = "classification&regression"


class RunMode(Enum):
    TRAIN = "train"
    VALID = "valid"
    # Test is not used for now
    TEST = "test"


# Emotion mapper to index
emotion2idx = {
    "surprise": 0,
    "fear": 1,
    "angry": 2,
    "neutral": 3,
    "happy": 4,
    "sad": 5,
    "disgust": 6,
    "disqust": 6,
}

# idx mapper to emotion
idx2emotion = {
    0: 'surprise',
    1: 'fear',
    2: 'angry',
    3: 'neutral',
    4: 'happy',
    5: 'sad',
    6: 'disgust'
}

# Gender Mapper
gender2idx = {
    "M": 0,
    "F": 1,
}

# Column names to be chosen from annotations .csv
columns_kemdy19 = OrderedDict({
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

columns_kemdy20 = OrderedDict({
    0: "Numb",
    1: "wav_start",
    2: "wav_end",
    3: "segment_id",
    4: "emotion",
    5: "valence",
    6: "arousal",
})

# emotion = (Valence, Arousal)
emotion_va_19_20_dict = {
    '0_centroid': (2.7013428, 3.6017902),
    '1_centroid': (1.8056872, 3.5810425),
    '2_centroid': (1.8337096, 3.7831194),
    '3_centroid': (2.989782, 3.1387265),
    '4_centroid': (4.093699, 3.7420354),
    '5_centroid': (1.9814088, 2.6549654),
    '6_centroid': (2.2050884, 3.1471236)
}

emotion_va_19_dict = {
    '0_centroid': (2.637565, 3.637669),
    '1_centroid': (1.7393138, 3.6382587),
    '2_centroid': (1.8028255, 3.8001664),
    '3_centroid': (2.971001, 2.981144),
    '4_centroid': (4.3137026, 3.8631923),
    '5_centroid': (1.895839, 2.6122148),
    '6_centroid': (2.205371, 3.1242967)
}

emotion_va_20_dict = {
    '0_centroid': (3.0942307, 3.380769),
    '1_centroid': (2.3906977, 3.0767443),
    '2_centroid': (2.2208333, 3.5694444),
    '3_centroid': (2.9960432, 3.1912591),
    '4_centroid': (3.838546, 3.6015217),
    '5_centroid': (2.5082645, 2.918182),
    '6_centroid': (2.2032788, 3.2934425)
}