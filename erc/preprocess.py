from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from erc.utils import get_logger
from erc.constants import selected_columns


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


def make_total_df(
    base_path: str | Path = "./data/KEMDy19",
    save_path: str | Path = "./data/kemdy19.csv",
) -> pd.DataFrame:
    """ Merges all .csv files to create integrated single dataframe for all segments and sessions.
    Iterates `base_path` annotation directory and reads in seperated csvs.
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
    
    logger.info(f"New dataframe saved as {save_path}")
    total_df.columns = list(selected_columns.values())
    total_df.to_csv(save_path, index=False)
    return total_df
