import os
import logging
from pathlib import Path
from collections import OrderedDict

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class KEMDy19Dataset(Dataset):
    wav_txt_path_fmt = "./data/KEMDy19/wav/Session{0}/Sess{0}_{1}"
    eda_path_fmt = "./data/KEMDy19/EDA/Session{0}/Original/Sess{0}{1}.csv"

    # Annotation (ECG / EDA / Emotion / Valence & Arousal)
    male_annot_expr = "./annotation/Session*_M_*"
    female_annot_expr = "./annotation/Session*_F_*"
    TOTAL_DF_PATH = "./data/kemdy19.csv"

    def __init__(self, base_path: str, generate_csv: bool):
        """
        Args:
            cfg: yaml file
        """
        logger.info("Instantiate KEMDy19 Dataset")
        self.base_path: Path = Path(base_path)
        self.total_df: pd.DataFrame = self.processed_db(generate_csv=generate_csv)

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

    def processed_db(self, generate_csv: bool = False) -> pd.DataFrame:
        """ Reads in .csv file if exists.
        If pre-processed .csv file does NOT exists, read from data path. """
        if not os.path.exists(self.TOTAL_DF_PATH) or generate_csv:
            logger.info(f"{self.TOTAL_DF_PATH} does not exists. Process from raw data")
            total_df = self.make_total_df()
        else:
            try:
                total_df = pd.read_csv(self.TOTAL_DF_PATH)
            except pd.errors.EmptyDataError as e:
                logger.error(f"{self.TOTAL_DF_PATH} seems to be empty")
                logger.exception(e)
                total_df = None
        return total_df

    def make_total_df(self) -> pd.DataFrame:
        # .csv 상태가 나빠서 위치로 기억하는 것이 나음
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
            "Segment ID": "segmend_id",
            "Total Evaluation": "emotion",
            "Unnamed: 11": "valence",
            "Unnamed: 12": "arousal",
        })
        male_annots = sorted(self.base_path.rglob(self.male_annot_expr))
        female_annots = sorted(self.base_path.rglob(self.female_annot_expr))
        assert (len(male_annots) > 0) & (len(female_annots) > 0),\
            f"Annotations does not exists. Check {self.base_path / self.male_annot_expr}"

        total_df = pd.DataFrame()
        pbar = tqdm(
            iterable=zip(male_annots, female_annots),
            desc="Processing ECG / EDA / Label",
            total=len(male_annots)
        )
        for m_ann, f_ann in pbar:
            m_df = pd.read_csv(m_ann).dropna()[list(selected_columns.keys())]
            f_df = pd.read_csv(f_ann).dropna()[list(selected_columns.keys())]

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
        
        logger.info(f"New dataframe saved as {self.TOTAL_DF_PATH}")
        total_df.columns = list(selected_columns.values())
        total_df.to_csv(self.TOTAL_DF_PATH, index=False)
        return total_df

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
