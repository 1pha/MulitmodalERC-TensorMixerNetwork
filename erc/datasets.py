from glob import glob
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class KEMDy19Dataset(Dataset):
    wav_txt_path_fmt = "./data/KEMDy19/wav/Session{0}/Sess{0}_{1}"
    eda_path_fmt = "./data/KEMDy19/EDA/Session{0}/Original/Sess{0}{1}.csv"

    # Annotation (ECG / EDA / Emotion / Valence & Arousal)
    MALE_ANN_PATH = "{}/annotation/Session*_M_*"
    FEMALE_ANN_PATH = "{}/annotation/Session*_F_*"
    TOTAL_DF_PATH = "./kemdy19.csv"

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: yaml file
        """
        self._base_path: str = cfg["BASE_PATH"]
        self.total_df: pd.DataFrame = self.processed_db(self)

    def __len__(self):
        return len(self.total_df)

    def __getitem__(self, idx):
        row = self.total_df.iloc[idx]
        session, script_type, speaker = row["Segment ID"].split("_")

        label = self.str2num(row["Total Evaluation"])
        wav_txt_path = self.wav_txt_path_fmt.format(session[-2:], script_type)
        # wav
        # 'KEMDy19/wav/Session01/Sess01_impro01' # wav, txt
        eda_path = self.eda_path_fmt.format(session[-2:], speaker[0])

        X = torch.tensor(self.X[idx], dtype=torch.float)
        label = torch.tensor([label], dtype=torch.long)

        sample = {"wav": None, "txt": None, "eda": None, "temp": None, "label": label}
        return sample

    @property
    def processed_db(self):
        if not isinstance(self.TOTAL_DF_PATH, Path):
            self.make_total_df()
        return pd.read_csv(self.TOTAL_DF_PATH)

    def make_total_df(self) -> None:
        selected_columns = [
            "Numb",
            "Wav",
            "Unnamed: 2",
            "ECG",
            "Unnamed: 4",
            "E4-EDA",
            "Unnamed: 6",
            "E4-TEMP",
            "Unnamed: 8",
            "Segment ID",
            "Total Evaluation",
        ]
        male_annotations = sorted(glob(self.MALE_ANN_PATH.format(self._base_path)))
        female_annotations = sorted(glob(self.FEMALE_ANN_PATH.format(self._base_path)))

        total_df = pd.DataFrame()
        for m_ann, f_ann in zip(male_annotations, female_annotations):
            m_df = pd.read_csv(m_ann).dropna()[selected_columns]
            f_df = pd.read_csv(f_ann).dropna()[selected_columns]

            # Sess01_impro03, Sess01_impro04의 TEMP와 E4-EDA값이 결측
            m_df = m_df[
                ~(m_df["Segment ID"].str.contains("Sess01_impro03"))
                & ~(m_df["Segment ID"].str.contains("Sess01_impro04"))
            ]
            f_df = f_df[
                ~(f_df["Segment ID"].str.contains("Sess01_impro03"))
                & ~(f_df["Segment ID"].str.contains("Sess01_impro04"))
            ]

            # 각 발화의 성별에 대한 감정만 추출
            m_df = m_df[m_df["Segment ID"].str.contains("M")]
            f_df = f_df[f_df["Segment ID"].str.contains("F")]

            # 다시 합쳐서 정렬
            tmp_df = pd.concat([m_df, f_df], axis=0).sort_values("Numb")
            total_df = pd.concat([total_df, tmp_df], axis=0)
        total_df.to_csv(self.TOTAL_DF_PATH, index=False)

    @staticmethod
    def str2num(key) -> int:
        emotion2idx = {
            "surprise": 1,
            "fear": 2,
            "angry": 3,
            "neutral": 4,
            "happy": 5,
            "sad": 6,
            "disgust": 7,
        }
        return emotion2idx.get(key, 0)
