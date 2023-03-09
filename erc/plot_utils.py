from collections import defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from erc.constants import emotion2idx


def drawing_ellipse(total_dict: dict, 
                    title: str = 'Ellipse Sampling (All)',
                    color_palette = sns.color_palette("Set2")) -> None:
    """
    patches.Ellipse(
       xy = (5, 5), # xy xy coordinates of ellipse centre.
       width = 5,   # width Total length (diameter) of horizontal axis.
       height = 10, # height Total length (diameter) of vertical axis.
       angle = -40, # angle Rotation in degrees anti-clockwise. 0 by default
       edgecolor = 'black',
       linestyle = 'solid', 
       fill = True,
       facecolor = 'yellow',
     ))
    """
    _, ax = plt.subplots(figsize=(5, 5))
    for idx, emotion in enumerate(total_dict.keys()):
        tmp_dict = total_dict[emotion]
        x,y, width, height = tmp_dict.values()

        ell = Ellipse(xy=(x, y),
                      width=width,
                      height=height,
                      color=color_palette[idx],
                      fill=False,
                      label=emotion,
                      alpha=0.9)
        ax.add_artist(ell)
    ax.set_xlim((1, 5))
    ax.set_ylim((1, 5))
    ax.legend()
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    plt.title(title)
    plt.grid()
    plt.show()


def split_df_by_gender(df: pd.DataFrame, total:bool = True):
    """
    Args:
        total: True all gender, False sperate into Female Male 
    """
    if total:
        return generate_eva_dict(df)
    else:
        male_dict = generate_eva_dict(df[df['gender'] == 0])
        female_dict = generate_eva_dict(df[df['gender'] == 1])
        return male_dict, female_dict


def generate_eva_dict(df_: pd.DataFrame) -> dict:
    """
    Summary: 
        감정 별 Valence Arousal mean, std 
    """
    idx2emotion = {v:k for k,v in emotion2idx.items()}
    choice_col = ['valence','arousal']
    mean_std_dict = defaultdict(dict)
    for emotion_idx in df_['emotion'].unique():
        means = df_[df_['emotion'] == emotion_idx][choice_col].mean(axis=0).values
        stds = df_[df_['emotion'] == emotion_idx][choice_col].std(axis=0).values

        tmp_dict = {
            'valence_mean': means[0],
            'arousal_mean' : means[1],
            'valence_std' : stds[0],
            'arousal_std': stds[1]
        }

        mean_std_dict[idx2emotion.get(emotion_idx)] = tmp_dict
    return mean_std_dict
