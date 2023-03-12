import os
from collections.abc import Iterable
from pathlib import Path
from typing import Tuple

import datasets
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import AutoProcessor, AutoTokenizer

from erc.preprocess import get_folds, merge_csv_kemdy19, merge_csv_kemdy20, run_generate_datasets
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
        mode: RunMode | str = RunMode.TRAIN,
        num_data: int = None,
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
        logger.debug("Instantiate %s Dataset", self.NAME)
        self.base_path: Path = Path(base_path)
        self.return_bio = return_bio
        self.max_length_wav = max_length_wav
        self.max_length_txt = max_length_txt
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer_name else None
        # This assertion is subject to change: number of folds to split
        assert isinstance(validation_fold, int) and validation_fold in range(-1, 5),\
            f"Validation fold should lie between 0 - 4, int. Given: {validation_fold}"
        self.validation_fold = validation_fold
        self.mode = RunMode[mode.upper()] if isinstance(mode, str) else mode
        self.df: pd.DataFrame = self.processed_db(generate_csv=generate_csv,
                                                  fold_num=validation_fold)
        
        # Limit number of data for debug (Fast Dev)
        if isinstance(num_data, int):
            if num_data in range(0, len(self.df)):
                self.num_data = num_data
            else:
                self.num_data = round(0.05 * len(self.df))
        else:
            self.num_data = None

    def __len__(self):
        return len(self.df) if not self.num_data else len(self.df[:self.num_data])

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
        row = self.df.iloc[idx]
        segment_id = row["segment_id"]
        data = {"segment_id": segment_id}
        _, _, gender, wav_prefix = self.parse_segment_id(segment_id=segment_id)
        
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
        data, sampling_rate = torchaudio.load(wav_path)
        if self.max_length_wav:
            # If self.max_length_wav is given, return a padded value
            # Else, just return naive wav file.
            data, mask = self.pad_value(data.squeeze(), max_length=self.max_length_wav)
        else:
            data = data.squeeze()
            mask = None
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
        if self.tokenizer:
            result: dict = self.tokenizer(text=txt,
                                        padding="max_length",
                                        truncation="only_first",
                                        max_length=self.max_length_txt,
                                        return_attention_mask=True,
                                        return_tensors="pt")
            input_ids = result["input_ids"].squeeze()
            mask = result["attention_mask"].squeeze()
            return input_ids, mask
        else:
            return txt, None

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
        emotion = emotion2idx.get(key, -1)
        return torch.tensor(emotion, dtype=torch.long)
    
    def gender2num(self, key: str) -> torch.Tensor:
        gender = gender2idx.get(key, -1)
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
        mode: RunMode | str = RunMode.TRAIN,
        num_data: int = None,
    ):
        super(KEMDy19Dataset, self).__init__(
            base_path,
            generate_csv,
            return_bio,
            max_length_wav,
            max_length_txt,
            tokenizer_name,
            validation_fold,
            mode,
            num_data,
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
        mode: RunMode | str = RunMode.TRAIN,
        num_data: int = None,
    ):
        super(KEMDy20Dataset, self).__init__(
            base_path,
            generate_csv,
            return_bio,
            max_length_wav,
            max_length_txt,
            tokenizer_name,
            validation_fold,
            mode,
            num_data,
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
        max_length_wav: int = 80_000,
        max_length_txt: int = 50,
        tokenizer_name: str = "klue/bert-base",
        mode: RunMode | str = RunMode.TRAIN,
        num_data: int = None,
    ):
        logger.info("Instantiate %s Dataset", mode)
        self.kemdy19 = KEMDy19Dataset(return_bio=return_bio,
                                      max_length_wav=max_length_wav,
                                      max_length_txt=max_length_txt,
                                      tokenizer_name=tokenizer_name,
                                      validation_fold=validation_fold,
                                      mode=mode,
                                      num_data=num_data)
        self.kemdy20 = KEMDy20Dataset(return_bio=return_bio,
                                      max_length_wav=max_length_wav,
                                      max_length_txt=max_length_txt,
                                      tokenizer_name=tokenizer_name,
                                      validation_fold=validation_fold,
                                      mode=mode,
                                      num_data=num_data)

    def __len__(self):
        return len(self.kemdy19) + len(self.kemdy20)

    def __getitem__(self, idx):
        if idx < len(self.kemdy19):
            return self.kemdy19.__getitem__(idx)
        else:
            return self.kemdy20.__getitem__(idx - len(self.kemdy19))
        

class HF_KEMD:
    def __init__(
        self,
        paths: list | str = ["kemdy19", "kemdy20"],
        validation_fold: int = 4,
        mode: RunMode | str = RunMode.TRAIN,
        wav_processor: str = "kresnik/wav2vec2-large-xlsr-korean",
        sampling_rate: int = 16_000,
        wav_max_length: int = 112_000, # 16_000 * 7, 7secs duration
        txt_processor: str = "klue/bert-base",
        txt_max_length: int = 64,
        load_from_cache_file: bool = True,
        num_proc: int = 8,
    ):
        """ Loads dataset and do pre-process
        Trunctae wav & text to designated maximum length
        Save to cache directory.
        Load from cache when cache_file_name is available.
        This is required since a full run takes around 20 minutes with no `num_proc`
        With `num_proc=8`, the whole process takes around 3 minutes.
        TODO: Check with DDP / Accelerate
        """
        # This assertion is subject to change: number of folds to split
        assert isinstance(validation_fold, int) and validation_fold in range(-1, 5),\
            f"Validation fold should lie between 0 - 4, int. Given: {validation_fold}"
        self.validation_fold = validation_fold
        self.mode = mode

        ds: datasets.arrow_dataset.Dataset = self.load_dataset(paths=paths)
        # Wave Processor
        self.wav_processor = AutoProcessor.from_pretrained(wav_processor) if wav_processor else None
        self.wav_kwargs = dict(
            sampling_rate=sampling_rate,
            max_length=wav_max_length,
            truncation="only_first",
            padding="max_length",
            padding_value=0,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Text Tokenizer
        self.txt_processor = AutoTokenizer.from_pretrained(txt_processor) if txt_processor else None
        self.txt_kwargs = dict(
            max_length=txt_max_length,
            truncation="only_first",
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Pre-process
        map_kwargs = dict(
            batched=True,
            desc="Pre-process wave & text",
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
        )
        self.ds = ds.map(self.preprocess, **map_kwargs).with_format("torch")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return self.ds[idx]

    def preprocess(self, batch: list):
        """ Mapping function for hf dataset """
        wav = self.wav_processor(audio=batch["wav"], **self.wav_kwargs)
        batch["wav"] = wav["input_values"]
        batch["wav_mask"] = wav["attention_mask"]

        txt = self.txt_processor(text=batch["txt"], **self.txt_kwargs)
        batch["txt"] = txt["input_ids"]
        batch["txt_mask"] = txt["attention_mask"]
        return batch
        
    def _load_dataset(self, path: str | Path):
        """ Loads huggingface dataset saved in .arr(feather) 
        If dataset does not exists, generate a new dataset. """
        if os.path.exists(path):
            logger.info("Load from disk, %s", path)
            ds = datasets.load_from_disk(path)
        else:
            logger.info("HF Dataset not found. Newly save from scratch. Path: %s", path)
            ds = run_generate_datasets(dataset_name=path)
        if self.validation_fold > 0:
            self.NUM_FOLDS = 5
            self.NUM_SESSIONS = {"kemdy19": 20, "kemdy20": 40}[path]
            ds = ds.filter(self.split_folds, input_columns=["id"], load_from_cache_file=False)
        return ds

    def split_folds(self, _id: list):
        fold_dict: dict = get_folds(num_session=self.NUM_SESSIONS, num_folds=self.NUM_FOLDS)
        fold_range: range = fold_dict[self.validation_fold]
        _id = int(_id.split("_")[0][-2:])
        include: bool = not (_id in (fold_range)) if self.mode == RunMode.TRAIN else \
                        _id in (fold_range)
        return include
            
    def load_dataset(self, paths):
        if isinstance(paths, str | Path):
            # Single data given
            ds = self._load_dataset(path=paths)
        elif isinstance(paths, Iterable):
            ds = datasets.concatenate_datasets([self._load_dataset(path) for path in paths])
        else:
            logger.warn("Wrongly given dataset. %s", paths)
            raise FileNotFoundError
        return ds


if __name__=="__main__":
    dataset = KEMDy20Dataset(base_path="./data/KEMDy20_v1_1",
                             generate_csv=True,
                             return_bio=False,
                             validation_fold=4,
                             mode="train")
    print(dataset[0])