from collections import defaultdict
import os 
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch 
from datasets import Dataset, load_from_disk

import erc
from erc.utils import get_logger
from erc.constants import columns_kemdy19, columns_kemdy20, emotion2idx


logger = get_logger(name=__name__)


def get_folds(num_session: int = 20, num_folds = 5) -> dict:
    """ Return a sequential fold split information
    For KEMDy19: 20 sessions
    For KEMDy20_v_1_1: 40 sessions """
    ns, div = num_session, num_folds
    num_sessions: list = [ns // div + (1 if x < ns % div else 0)  for x in range (div)]
    fold_dict = dict()
    for f in range(num_folds):
        s = sum(num_sessions[:f])
        e = s + num_sessions[f]
        fold_dict[f] = range(s + 1, e + 1) # Because sessions starts from 1
    return fold_dict


def eda_preprocess(file_path: str) -> pd.DataFrame:
    """ on_bad_line이 있어서 (column=4 or  3으로 일정하지 않아) 4줄로 통일 하는 함수 """
    columns = ["EDA_value", "a", "b", "Segment ID"]

    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in tqdm(lines):
        line = line.rstrip()
        if len(line.split(",")) <= 3:
            line += ",None"  # 4줄로 만들어주기 .
        new_lines.append(line.split(","))

    return pd.DataFrame(new_lines, columns=columns).replace("None", np.nan).dropna()


def map_emotion(arr):
    return np.vectorize(erc.constants.emotion2idx.get)(arr)


def merge_csv_kemdy19(
    base_path: str | Path = "./data/KEMDy19",
    save_path: str | Path = "./data/kemdy19.csv",
    exclude_multilabel: bool = True,
) -> pd.DataFrame:
    """ Merges all .csv files to create integrated single dataframe for all segments and sessions.
    Iterates `base_path` annotation directory and reads in seperated csvs.
    Note that this will always overwrite csv file on save_path (no handlings)
    """
    # Annotation (ECG / EDA / Emotion / Valence & Arousal)
    male_annot_expr = "./annotation/Session*_M_*"
    female_annot_expr = "./annotation/Session*_F_*"

    base_path, save_path = Path(base_path), Path(save_path)

    male_annots = sorted(base_path.rglob(male_annot_expr))
    female_annots = sorted(base_path.rglob(female_annot_expr))
    assert (len(male_annots) > 0) & (len(female_annots) > 0),\
        f"Annotations does not exists. Check {base_path / male_annot_expr}"

    total_df = pd.DataFrame()
    pbar = tqdm(
        iterable=zip(male_annots, female_annots),
        desc=f"Processing ECG / EDA / Label from {base_path}",
        total=min(len(male_annots), len(female_annots))
    )
    emo_keys = list(emotion2idx.keys())
    emo_keys.remove("disqust")
    for m_ann, f_ann in pbar:
        m_df = pd.read_csv(m_ann).dropna()
        f_df = pd.read_csv(f_ann).dropna()
        col_filter = list(columns_kemdy19.keys())
        if not exclude_multilabel:
            m_emo, f_emo = m_df.iloc[:, -30::3], f_df.iloc[:, -30::3]
            for emo, idx in emotion2idx.items():
                if emo == "disqust":
                    emo = "disgust"
                m_df[emo] = (map_emotion(m_emo) == idx).sum(axis=1)
                f_df[emo] = (map_emotion(f_emo) == idx).sum(axis=1)
            col_filter += emo_keys
            assert len(col_filter) == 20, f"# cols should be 20: existing 13 + 7 emotions"
        m_df = m_df[col_filter]
        f_df = f_df[col_filter]

        # Sess01_impro03, Sess01_impro04의 TEMP와 E4-EDA값이 결측
        # 다른 Session에서도 결측값은 있으나, 해당 두 세션에는 결측값이 너무 많아 유효한 데이터가 아니라고 판단하여
        # 모두 사용하지 않기로함
        _drop_scripts = ['Sess01_impro03', 'Sess01_impro04', 'Sess03_script04', 'Sess06_script02', 'Sess07_script02']
        m_df = m_df[~(m_df["Segment ID"].str.contains('|'.join(_drop_scripts)))]
        f_df = f_df[~(f_df["Segment ID"].str.contains('|'.join(_drop_scripts)))]

        # 각 발화의 성별에 대한 감정만 추출
        m_df = m_df[m_df["Segment ID"].str.contains("M")]
        f_df = f_df[f_df["Segment ID"].str.contains("F")]

        # 다시 합쳐서 정렬
        tmp_df = pd.concat([m_df, f_df], axis=0).sort_values("Numb")
        total_df = pd.concat([total_df, tmp_df], axis=0)
    
    total_df.columns = list(columns_kemdy19.values()) + (emo_keys if not exclude_multilabel else [])
    if exclude_multilabel:
        total_df = total_df[~total_df['emotion'].str.contains(';')]
    else:
        # Check missing emotion
        assert (total_df.iloc[:, -7:].sum(axis=1) == 10).sum() == total_df.shape[0],\
            f"Make sure there is no mislabeled emotion{total_df}"

    if save_path:
        total_df.to_csv(save_path, index=False)
        logger.info(f"New dataframe saved as {save_path}")
        logger.info(f"Created dataframe has shape of {total_df.shape}")
    return total_df


def merge_csv_kemdy20(
    base_path: str | Path = "./data/KEMDy20_v1_1",
    save_path: str | Path = "./data/kemdy20.csv",
    exclude_multilabel: bool = True,
) -> pd.DataFrame:
    """ Merges all .csv files to create integrated single dataframe for all segments and sessions.
    Iterates `base_path` annotation directory and reads in seperated csvs.
    Note that this will always overwrite csv file on save_path (no handlings)
    
    Since behavior of two datasets KEMDy19 and KEMDy20_v1_1 is different,
    function was unfortuantely diverged.
    """
    # Annotation (ECG / EDA / Emotion / Valence & Arousal)
    base_path, save_path = Path(base_path), Path(save_path)
    annot_fmt = "annotation/*.csv"
    annots = sorted(base_path.rglob(annot_fmt))
    assert len(annots) > 0, f"Annotations does not exists. Check {base_path / annot_fmt}"

    total_df = pd.DataFrame()
    pbar = tqdm(
        iterable=annots,
        desc=f"Processing {base_path}",
        total=len(annots)
    )
    emo_keys = list(emotion2idx.keys())
    emo_keys.remove("disqust")
    col_filter = list(columns_kemdy20.keys())
    for ann in pbar:
        df = pd.read_csv(ann).dropna().iloc[1:]
        if not exclude_multilabel:
            emo_df = df.iloc[:, -30::3]
            for emo, idx in emotion2idx.items():
                if emo == "disqust":
                    emo = "disgust"
                df[emo] = (map_emotion(emo_df) == idx).sum(axis=1)
            df = pd.concat([df.iloc[:, col_filter], df.iloc[:, -7:]], axis=1)
        else:
            df = df.iloc[:, col_filter]
        total_df = pd.concat([total_df, df], axis=0).sort_values("Numb")

    total_df.columns = list(columns_kemdy20.values()) + (emo_keys if not exclude_multilabel else [])
    if exclude_multilabel:
        total_df = total_df[~total_df['emotion'].str.contains(';')]
    else:
        # Check missing emotion
        assert (total_df.iloc[:, -7:].sum(axis=1) == 10).sum() == total_df.shape[0],\
            f"Make sure there is no mislabeled emotion{total_df}"
    if save_path:
        total_df.to_csv(save_path, index=False)
        logger.info(f"New dataframe saved as {save_path}")
        logger.info(f"Created dataframe has shape of {total_df.shape}")
    return total_df


def generate_datasets(
    dataset_: torch.utils.data.Dataset, 
    save_name: str = 'audio_dataset_19',
    mode: str = 'train',
    validation_fold: int = -1,
    overrides: bool = False,
    exclude_columns: list = (),
) -> Dataset:
    # select columns
    column_dict = {
        "segment_id": "id",
        "wav": "wav",
        "txt": "txt",
        "emotion": "label",
        "valence": "valence",
        "arousal": "arousal",
        "gender": "gender,"
    }
    for e in exclude_columns:
        column_dict.pop(e)
    save_name = os.path.join(save_name,
                             f'{mode}_{validation_fold:02d}' if validation_fold > 0 else "")

    total_train_dataset_dict, num_error_cases = defaultdict(list), set()
    # Check existence
    if (os.path.exists(save_name) == False) or (overrides == True):
        for data in tqdm(iterable=dataset_):
            if len(data) == 1:
                # Error cases. Do not save
                num_error_cases.add(data["segment_id"])
                continue
            for fetch_key, save_key in column_dict.items():
                total_train_dataset_dict[save_key].append(data[fetch_key])

        # Generate dataset from dict and save 
        ds = Dataset.from_dict(total_train_dataset_dict)
        ds.save_to_disk(save_name)
    else:
        ds = load_from_disk(save_name).with_format("torch")
    return ds


def run_generate_datasets(dataset_name="kemdy19"):
    """ Loadds torch dataset and dump it into huggingface dataset
    This is utilize huggingface datasets' `map` method.
    This function will override existing datasets."""

    _ds_kwargs = {
        "tokenizer_name": None, "max_length_wav": None, "validation_fold": -1
    }
    dataset_: torch.utils.data.Dataset = {
        "kemdy19": erc.datasets.KEMDy19Dataset(**_ds_kwargs),
        "kemdy20": erc.datasets.KEMDy20Dataset(**_ds_kwargs),
    }[dataset_name]
    ds = generate_datasets(dataset_=dataset_,
                           validation_fold=-1,
                           save_name=dataset_name,
                           overrides=True,)
    return ds
