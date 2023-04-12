from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
try:
    # Installing irrCAC provokes an error in unknown cases
    # Since this is not explicitly used in baseline, we exclude this for now
    from irrCAC.raw import CAC
except:
    pass

from erc.constants import emotion2idx



def get_eval_group(df: pd.DataFrame) -> set:
    cols = df.iloc[:, -30:].columns
    cols = set(filter(lambda s: s.startswith("Eval"), cols))
    return cols


def kemdy19_get_rater(
    base_path: str | Path = "./data/KEMDy19",
    save_path: str | Path = "./data/kemdy19.csv",
) -> Dict[int, pd.DataFrame]:
    # Annotation (ECG / EDA / Emotion / Valence & Arousal)
    male_annot_expr = "./annotation/Session*_M_*"
    female_annot_expr = "./annotation/Session*_F_*"

    base_path, save_path = Path(base_path), Path(save_path)

    male_annots = sorted(base_path.rglob(male_annot_expr))
    female_annots = sorted(base_path.rglob(female_annot_expr))
    assert (len(male_annots) > 0) & (len(female_annots) > 0),\
        f"Annotations does not exists. Check {base_path / male_annot_expr}"

    pbar = tqdm(
        iterable=zip(male_annots, female_annots),
        desc=f"Processing ECG / EDA / Label from {base_path}",
        total=min(len(male_annots), len(female_annots))
    )
    emo_keys = list(emotion2idx.keys())
    emo_keys.remove("disqust")
    
    rater_groups = dict()
    grouped_df = dict()
    group_num = 0
    for m_ann, f_ann in pbar:
        m_df = pd.read_csv(m_ann).dropna()
        f_df = pd.read_csv(f_ann).dropna()
        
        found = False
        cur_group = get_eval_group(m_df)
        for k, v in rater_groups.items():
            found = v == cur_group
            if found:
                group_num = k
                break
        if found is False:
            group_num += 1
            rater_groups[group_num] = cur_group
        
        _drop_scripts = ['Sess01_impro03', 'Sess01_impro04', 'Sess03_script04', 'Sess06_script02', 'Sess07_script02']
        m_df = m_df[~(m_df["Segment ID"].str.contains('|'.join(_drop_scripts)))]
        f_df = f_df[~(f_df["Segment ID"].str.contains('|'.join(_drop_scripts)))]

        # 각 발화의 성별에 대한 감정만 추출
        m_df = m_df[m_df["Segment ID"].str.contains("M")]
        f_df = f_df[f_df["Segment ID"].str.contains("F")]

        # 다시 합쳐서 정렬
        tmp_df = pd.concat([m_df, f_df], axis=0).sort_values("Numb").reset_index(drop=True)
        grouped_df[group_num] = pd.concat([grouped_df.get(group_num, pd.DataFrame()), tmp_df], axis=0)\
                                .sort_values(by="Numb")\
                                .reset_index(drop=True)
        
    grouped_dict_df = dict()
    for gn, _df in grouped_df.items():
        _df = {
            "segment_id": _df["Segment ID"],
            "emotion": _df.iloc[:, -30::3],
            "valence": _df.iloc[:, -29::3].astype(float),
            "arousal": _df.iloc[:, -28::3].astype(float),
        }
        _df["arousal"].columns = _df["emotion"].columns
        _df["valence"].columns = _df["emotion"].columns
        grouped_dict_df[gn] = _df
    return grouped_dict_df


def kemdy20_get_rater(
    base_path: str | Path = "./data/KEMDy20_v1_1",
    save_path: str | Path = "./data/kemdy20.csv",
) -> Dict[int, pd.DataFrame]:
    # Annotation (ECG / EDA / Emotion / Valence & Arousal)
    base_path, save_path = Path(base_path), Path(save_path)
    annot_fmt = "annotation/*.csv"
    annots = sorted(base_path.rglob(annot_fmt))
    assert len(annots) > 0, f"Annotations does not exists. Check {base_path / annot_fmt}"

    pbar = tqdm(
        iterable=annots,
        desc=f"Processing {base_path}",
        total=len(annots)
    )
    emo_keys = list(emotion2idx.keys())
    emo_keys.remove("disqust")
    
    rater_groups = dict()
    grouped_df = dict()
    group_num = 0
    for ann in pbar:
        df = pd.read_csv(ann).dropna().iloc[1:]
        
        found = False
        cur_group = get_eval_group(df)
        for k, v in rater_groups.items():
            found = v == cur_group
            if found:
                group_num = k
                break
        if found is False:
            group_num += 1
            rater_groups[group_num] = cur_group
        
        grouped_df[group_num] = pd.concat([grouped_df.get(group_num, pd.DataFrame()), df], axis=0)\
                                .sort_values(by="Numb")\
                                .reset_index(drop=True)
    
    grouped_dict_df = dict()
    for gn, _df in grouped_df.items():
        _df = {
            "segment_id": _df["Segment ID"],
            "emotion": _df.iloc[:, -30::3],
            "valence": _df.iloc[:, -29::3].astype(float),
            "arousal": _df.iloc[:, -28::3].astype(float),
        }
        _df["arousal"].columns = _df["emotion"].columns
        _df["valence"].columns = _df["emotion"].columns
        grouped_dict_df[gn] = _df
    return grouped_dict_df


def get_corr_mean(corr: dict, label: str = "arousal") -> np.ndarray:
    corr: np.ndarray = corr[label].values
    np.fill_diagonal(a=corr, val=np.nan)
    mean = np.nanmean(corr, axis=0)
    std = np.nanstd(corr, axis=0)
    return mean, std


def _plot_heatmap(df: dict,
                  col: str,
                  ax: plt.axes,
                  group_name: str = None,
                  mask: pd.Series | str = None) -> pd.DataFrame:
    if mask is not None:
        if isinstance(mask, pd.Series):
            df = df[col][mask]
        elif isinstance(mask, str):
            _mask = df["segment_id"].str.contains(mask)
            df = df[col][_mask]
    else:
        df = df[col]
    corr = df.astype(float).corr()
    _gtitle = f"{group_name if group_name is not None else ''} " + f"{mask if isinstance(mask, str) else ''}"
    title = f"{_gtitle} {col}"
    ax.set_title(title.capitalize())
    sns.heatmap(corr, ax=ax, vmin=-.2, vmax=1, annot=True, fmt=".2f");
    return corr
    

def plot_heatmap(df: dict,
                 group_idx: int = None,
                 mask: pd.Series | str = None,
                 suptitle: str = "") -> dict:
    fig, ax = plt.subplots(figsize=(14, 6), ncols=2)
    group_name = "" if group_idx is None else f"Rater Group {group_idx}: "
    corrs = dict()
    for idx, col in enumerate(["arousal", "valence"]):
        corrs[col] = _plot_heatmap(df=df, col=col, ax=ax[idx], group_name=group_name, mask=mask)
    fig.suptitle(suptitle if suptitle else "")
    return corrs
    
    
def get_irr(df: dict | pd.DataFrame, label: str = "emotion", mask: pd.Series | str = None):
    if mask is not None:
        mask = df["segment_id"].str.contains(mask)
        df = df[label][mask]
    else:
        df = df[label]
    print(label)
    table = CAC(df)
    
    fleiss = table.fleiss()
    krippendorff = table.krippendorff()
    print(f"Fleiss Kappa: {fleiss['est']['coefficient_value']}", end="\t")
    print(f"Krippendorff: {krippendorff['est']['coefficient_value']}")
    return {
        "fleiss": fleiss,
        "krippendorff": krippendorff,
    }