import os 

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from collections import defaultdict

import torch 
from datasets import Dataset, load_from_disk

from erc.utils import get_logger
from erc.constants import columns_kemdy19, columns_kemdy20


logger = get_logger()


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


def merge_csv_kemdy19(
    base_path: str | Path = "./data/KEMDy19",
    save_path: str | Path = "./data/kemdy19.csv",
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
    for m_ann, f_ann in pbar:
        m_df = pd.read_csv(m_ann).dropna()[list(columns_kemdy19.keys())]
        f_df = pd.read_csv(f_ann).dropna()[list(columns_kemdy19.keys())]

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

    
    logger.info(f"New dataframe saved as {save_path}")
    total_df.columns = list(columns_kemdy19.values())
    total_df = total_df[~total_df['emotion'].str.contains(';')]
    total_df.to_csv(save_path, index=False)
    return total_df


def merge_csv_kemdy20(
    base_path: str | Path = "./data/KEMDy20_v1_1",
    save_path: str | Path = "./data/kemdy20.csv",
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
    for ann in pbar:
        df = pd.read_csv(ann).dropna().iloc[1:, list(columns_kemdy20.keys())]
        total_df = pd.concat([total_df, df], axis=0).sort_values("Numb")
    
    logger.info(f"New dataframe saved as {save_path}")
    total_df.columns = list(columns_kemdy20.values())
    total_df = total_df[~total_df['emotion'].str.contains(';')]
    total_df.to_csv(save_path, index=False)
    return total_df


def generate_datasets(
        dataset_ : torch.utils.data.Dataset, 
        save_name : str = 'audio_dataset_19',
        mode : str = 'train',
        validation_fold : int = 4,
        overrides : bool = False,
    ) -> Dataset:
    # select columns 
    origin_name = ['wav', 'wav_mask', 'emotion', 'valence', 'arousal', 'gender'][:3]
    convert_name = ['input_values', 'attention_mask', 'label','valence','arousal', 'gender'][:3]
    save_name = os.path.join(save_name, f'{mode}_{validation_fold:02d}')
    
    total_train_dataset_dict = defaultdict(list)

    # check existance 
    if (os.path.exists(save_name) == False) or (overrides==True):

        pbar = tqdm(total = len(dataset_)+1)
        for idx, batch in enumerate(dataset_):
            total_train_dataset_dict["id"].append(idx) # give primary key_
            for key_, c_key_ in zip(origin_name, convert_name):
                
                total_train_dataset_dict[c_key_].append(batch[key_])
            pbar.update(1)
        
        # generage dataset from dict and save 
        ds = Dataset.from_dict(total_train_dataset_dict)
        ds.save_to_disk(save_name)
                        
    else:
        ds = load_from_disk(save_name).with_format("torch")
    return ds