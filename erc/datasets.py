import os
from pathlib import Path
from typing import Tuple

from scipy.io import wavfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import AutoTokenizer

from erc.preprocess import get_folds, merge_csv_kemdy19, merge_csv_kemdy20
from erc.utils import check_exists, get_logger
from erc.constants import RunMode, emotion2idx, gender2idx


logger = get_logger()


class KEMDBase(Dataset):
    """ Abstract class base for KEMD dataset """
    NUM_FOLDS = 5
    def __init__(
        self,
        base_path: str,
        generate_csv: bool = False,
        return_bio: bool = False,
        max_length_wav: int = 200_000,
        max_length_txt: int = 50,
        tokenizer_name: str = "klue/bert-base",
        validation_fold: int = 4,
        mode: RunMode | str = RunMode.TRAIN
    ):
        """
        Args:
            base_path:
                Only used when csv is not found. Optional
            generate_csv:
                Flag to generate a new label.csv, default=False
            return_bio:
                Flag to call and return full ECG / EDA / TEMP data
                Since csv in annotation directory contains start/end value of above signals,
                this is not necessary. default=False
            validation_fold:
                Indicates validation fold.
                Fold split is based on Session number
                i.e. 
                    - Fold 0: Session 1 - 4
                    - Fold 1: Session 5 - 8
                    - Fold 2: Session 9 - 12
                    - Fold 3: Session 13 - 16
                    - Fold 4: Session 17 - 20
                If -1 given, load the whole fold
            mode:
                Train / valid / test mode.
        """
        logger.info("Instantiate %s Dataset", self.NAME)
        self.base_path: Path = Path(base_path)
        self.return_bio = return_bio
        self.max_length_wav = max_length_wav
        self.max_length_txt = max_length_txt
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # This assertion is subject to change: number of folds to split
        assert isinstance(validation_fold, int) and validation_fold in range(-1, 5),\
            f"Validation fold should lie between 0 - 4, int. Given: {validation_fold}"
        self.validation_fold = validation_fold
        self.mode = RunMode[mode.upper()] if isinstance(mode, str) else mode

        self.df: pd.DataFrame = self.processed_db(generate_csv=generate_csv,
                                                  fold_num=validation_fold)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        """ Returns data dictionary.
        Element specifications
            sampling_rate:
                int (e.g. 16_000)
            wav:
                torch.int16
                ndim=1, (max_length_wav,)
            wav_mask:
                torch.int64
                ndim=1, (max_length_wav,)
            txt:
                torch.int64
                ndim=1, (max_length_txt,)
            txt_mask:
                torch.int64
                ndim=1, (max_length_txt,)
            emotion, gender:
                torch.int64
                ndim=0
            valence, arousal:
                torch.float32
                ndim=0
        """
        data = dict()
        row = self.df.iloc[idx]
        segment_id = row["segment_id"]
        session, speaker, gender, wav_prefix = self.parse_segment_id(segment_id=segment_id)
        
        wav_path = wav_prefix / f"{segment_id}.wav"
        txt_path = wav_prefix / f"{segment_id}.txt"
        if not os.path.exists(wav_path) or not os.path.exists(txt_path):
            # Pre-checking data existence
            # TODO This should be dealed! Not a good behavior
            logger.warn("Error occurs -> %s", wav_prefix)
            return data

        # Wave File
        sampling_rate, wav, wav_mask = self.get_wav(wav_path=wav_path)
        data["sampling_rate"] = sampling_rate
        data["wav"] = wav
        data["wav_mask"] = wav_mask
        
        # Txt File
        txt, txt_mask = self.get_txt(txt_path=txt_path, encoding=self.TEXT_ENCODING)
        data["txt"] = txt
        data["txt_mask"] = txt_mask
        
        # Bio Signals
        # Currently returns average signals across time elapse
        if self.return_bio:
            for bio in ["ecg", "e4-eda", "e4-temp"]:
                s, e = map(float, row[[f"{bio}_start", f"{bio}_end"]])
                data[bio] = torch.tensor((s + e) / 2, dtype=torch.float)
                
        # Emotion
        data["emotion"] = self.str2num(row["emotion"])

        # Valence & Arousal
        valence, arousal = map(float, row[["valence", "arousal"]])
        data["valence"] = torch.tensor(valence, dtype=torch.float)
        data["arousal"] = torch.tensor(arousal, dtype=torch.float)

        # Man-Female
        data["gender"] = self.gender2num(gender) # Sess01_script01_F003
        return data
    
    def pad_value(
        self,
        arr: torch.Tensor,
        max_length: int,
        pad_value: int | float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Assumes single data """
        if not isinstance(arr, torch.Tensor):
            arr = torch.tensor(arr)
        
        mask = torch.ones(max_length).long()
        if len(arr) >= max_length:
            arr = arr[:max_length]
        else:
            null_size = max_length - len(arr)
            arr = torch.nn.functional.pad(arr, pad=(0, null_size), value=pad_value)
            mask[null_size:] = 0
        return arr, mask

    def get_wav(self, wav_path: Path | str) -> torch.Tensor | np.ndarray:
        """ Get output feature vector from pre-trained wav2vec model
        XXX: Embedding outside dataset, to fine-tune pre-trained model? See Issue
        """
        wav_path = check_exists(wav_path)
        # sampling_rate, data = wavfile.read(wav_path)
        data, sampling_rate = torchaudio.load(wav_path)
        data, mask = self.pad_value(data.squeeze(), max_length=self.max_length_wav)
        return sampling_rate, data, mask

    def get_txt(self, txt_path: Path | str, encoding: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Get output feature vector from pre-trained txt model
        :parameters:
            txt_path: Path to text in Sessions
            encoding: For KEMDy19, None. For KEMDy20_v1_1, cp949
        TODO:
            1. How to process special cases: /o /l 
                -> They _maybe_ processed by pre-trained toeknizers
            2. Which model to use
        """
        txt_path = check_exists(txt_path)
        with open(txt_path, mode="r", encoding=encoding) as f:
            txt: list = f.readlines()
        # We assume there is a single line
        txt: str = " ".join(txt)
        result: dict = self.tokenizer(text=txt,
                                                 padding="max_length",
                                                 truncation="only_first",
                                                 max_length=self.max_length_txt,
                                                 return_attention_mask=True,
                                                 return_tensors="pt")
        input_ids = result["input_ids"].squeeze()
        mask = result["attention_mask"].squeeze()
        return input_ids, mask

    def processed_db(self, generate_csv: bool = False, fold_num: int = 4) -> pd.DataFrame:
        """ Reads in .csv file if exists.
        If pre-processed .csv file does NOT exists, read from data path. """
        if not os.path.exists(self.TOTAL_DF_PATH) or generate_csv:
            logger.info(f"{self.TOTAL_DF_PATH} does not exists. Process from raw data")
            total_df = self.merge_csv(base_path=self.base_path, save_path=self.TOTAL_DF_PATH)
        else:
            try:
                total_df = pd.read_csv(self.TOTAL_DF_PATH)
            except pd.errors.EmptyDataError as e:
                logger.error(f"{self.TOTAL_DF_PATH} seems to be empty")
                logger.exception(e)
                total_df = None
        
        df = self.split_folds(total_df=total_df, fold_num=fold_num, mode=self.mode)
        return df
    
    def split_folds(
        self,
        total_df: pd.DataFrame,
        fold_num: int,
        mode: RunMode | str = RunMode.TRAIN,
    ) -> pd.DataFrame:
        if fold_num == -1:
            return total_df
        else:
            sessions: pd.Series = total_df["segment_id"].apply(lambda s: s.split("_")[0][-2:])
            sessions = sessions.apply(int)
            fold_dict: dict = get_folds(num_session=self.NUM_SESSIONS, num_folds=self.NUM_FOLDS)
            fold_range: range = fold_dict[fold_num]
            loc = ~sessions.isin(fold_range) if mode == RunMode.TRAIN else sessions.isin(fold_range)
            return total_df.loc[loc]
    
    def str2num(self, key: str) -> torch.Tensor:
        emotion = emotion2idx.get(key, 0)
        return torch.tensor(emotion, dtype=torch.long)
    
    def gender2num(self, key: str) -> torch.Tensor:
        gender = gender2idx.get(key, 0)
        return torch.tensor(gender, dtype=torch.long)
    
    def parse_segment_id(self, segment_id: str) -> Tuple[str, str, str, str]:
        """ Parse `segment_id` into useful information 
        This varies across dataset. Needs to be implemented for `__getitem__` method """
        raise NotImplementedError

    def merge_csv(
        self,
        base_path: str | Path = "./data/KEMDy20_v1_1",
        save_path: str | Path = "./data/kemdy20.csv",
    ):
        """ Loads all annotation .csv and merge into a single csv.
        This function is called when target .csv is not found. """
        raise NotImplementedError
    

class KEMDy19Dataset(KEMDBase):
    NAME = "KEMDy19"
    WAV_PATH_FMT = "./data/KEMDy19/wav/Session{0}/Sess{0}_{1}"
    EDA_PATH_FMT = "./data/KEMDy19/EDA/Session{0}/Original/Sess{0}{1}.csv"

    NUM_SESSIONS = 20
    TOTAL_DF_PATH = "./data/kemdy19.csv"
    TEXT_ENCODING: str = None

    def __init__(
        self,
        base_path: str = "./data/KEMDy19",
        generate_csv: bool = False,
        return_bio: bool = True,
        max_length_wav: int = 200_000,
        max_length_txt: int = 50,
        tokenizer_name: str = "klue/bert-base",
        validation_fold: int = 4,
        mode: RunMode | str = RunMode.TRAIN
    ):
        super(KEMDy19Dataset, self).__init__(
            base_path,
            generate_csv,
            return_bio,
            max_length_wav,
            max_length_txt,
            tokenizer_name,
            validation_fold,
            mode
        )

    def merge_csv(
        self,
        base_path: str | Path = "./data/KEMDy19",
        save_path: str | Path = "./data/kemdy19.csv",
    ):
        return merge_csv_kemdy19(base_path=base_path, save_path=save_path)

    def parse_segment_id(self, segment_id: str) -> Tuple[str, str, str, str]:
        """ KEMDy19
            segment_id: Sess01_script01_M001
            prefix: 'KEMDy19/wav/Session01/Sess01_impro01'
        """
        session, script_type, speaker = segment_id.split("_")
        wav_prefix = Path(self.WAV_PATH_FMT.format(session[-2:], script_type))
        gender = speaker[0]
        return session, speaker, gender, wav_prefix


class KEMDy20Dataset(KEMDBase):
    NAME = "KEMDy20"
    WAV_PATH_FMT = "./data/KEMDy20_v1_1/wav/Session{0}"
    # Not used yet
    EDA_PATH_FMT = "./data/KEMDy20v_1_1/EDA/Session{0}/Original/Sess{0}{1}.csv"

    NUM_SESSIONS = 40
    TOTAL_DF_PATH = "./data/kemdy20.csv"
    TEXT_ENCODING: str = "cp949"

    def __init__(
        self,
        base_path: str = "./data/KEMDy20_v1_1",
        generate_csv: bool = False,
        return_bio: bool = False,
        max_length_wav: int = 200_000,
        max_length_txt: int = 50,
        tokenizer_name: str = "klue/bert-base",
        validation_fold: int = 4,
        mode: RunMode | str = RunMode.TRAIN
    ):
        super(KEMDy20Dataset, self).__init__(
            base_path,
            generate_csv,
            return_bio,
            max_length_wav,
            max_length_txt,
            tokenizer_name,
            validation_fold,
            mode
        )

    def merge_csv(
        self,
        base_path: str | Path = "./data/KEMDy20_v1_1",
        save_path: str | Path = "./data/kemdy20.csv",
    ):
        return merge_csv_kemdy20(base_path=base_path, save_path=save_path)

    def parse_segment_id(self, segment_id: str) -> Tuple[str, str, str, str]:
        """
        KEMDy20_v1_1
            segment_id: Sess01_script01_User002M_001
            prefix: 'KEMDy20_v1_1/wav/Session01'
        """
        session, _, speaker, _ = segment_id.split("_")
        wav_prefix = Path(self.WAV_PATH_FMT.format(session[-2:]))
        gender = speaker[-1]
        return session, speaker, gender, wav_prefix


class KEMDDataset(Dataset):
    """ Integrated dataset for KEMDy19 and KEMDy20_v1_1
    Example codes:
    ```config
    dataset:
        _target_: erc.datasets.KEMDDataset
        return_bio: False
        validation_fold: 4
        mode: train
    ```
    ```python
    with hydra.initialize(version_base=None, config_path="./config"):
        cfg = hydra.compose(config_name="config")
    dataset = hydra.utils.instantiate(cfg.dataset)
    dataloader = hydra.utils.instantiate(cfg.dataloader)
    batch = next(iter(dataloader))
    ```
    """
    def __init__(
        self,
        return_bio: bool = False,
        validation_fold: int = 4,
        max_length_wav: int = 200_000,
        max_length_txt: int = 50,
        tokenizer_name: str = "klue/bert-base",
        mode: RunMode | str = RunMode.TRAIN
    ):
        self.kemdy19 = KEMDy19Dataset(return_bio=return_bio,
                                      max_length_wav=max_length_wav,
                                      max_length_txt=max_length_txt,
                                      tokenizer_name=tokenizer_name,
                                      validation_fold=validation_fold,
                                      mode=mode)
        self.kemdy20 = KEMDy20Dataset(return_bio=return_bio,
                                      max_length_wav=max_length_wav,
                                      max_length_txt=max_length_txt,
                                      tokenizer_name=tokenizer_name,
                                      validation_fold=validation_fold,
                                      mode=mode)

    def __len__(self):
        return len(self.kemdy19) + len(self.kemdy20)

    def __getitem__(self, idx):
        if idx < len(self.kemdy19):
            return self.kemdy19.__getitem__(idx)
        else:
            return self.kemdy20.__getitem__(idx - len(self.kemdy19))


if __name__=="__main__":
    dataset = KEMDy20Dataset(base_path="./data/KEMDy20_v1_1",
                             generate_csv=True,
                             return_bio=False,
                             validation_fold=4,
                             mode="train")
    print(dataset[0])