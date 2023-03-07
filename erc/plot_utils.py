import pandas as pd

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse


def drawing_ellipse(total_dict:dict, 
                    title:str='Ellipse Sampling (All)',
                    color_palette=sns.color_palette("Set2")) -> None:
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
    fig, ax = plt.subplots(figsize=(5, 5))
    
    for idx, emotion in enumerate(total_dict.keys()):
        tmp_dict = total_dict[emotion]
        x,y, width, hight  = tmp_dict.values()

        tmp = Ellipse((x,y), width, hight,
                    color=color_palette[idx],
                    fill = False,
                    label=emotion,alpha=0.9) # Ellipse (x,y), width,  hight 
        ax.add_artist(tmp)
    ax.set_xlim((1, 5))
    ax.set_ylim((1, 5))
    ax.legend()
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    plt.title(title)
    plt.grid()
    plt.show()

def split_df_by_gender(df: pd.DataFrame, total:bool=True):
    """
    Args:
        total: True all gender , False sperate into Female Male 
    """
    if total:
        return generate_eva_dict(df)
    else:
        male_dict = generate_eva_dict(df[df['gender'] == 1])
        female_dict = generate_eva_dict(df[df['gender'] == 2])
        return male_dict, female_dict


def generate_eva_dict(df_: pd.DataFrame)->dict:
    """
    Summary: 
        감정 별 Valence Arousal mean, std 

    """
    emotion_r= { 
    1: 'surprise',
    2: 'fear',
    3: 'angry',
    4: 'neutral',
    5: 'happy',
    6: 'sad',
    7: 'disgust'}
    choice_col = ['valence','arousal']
    mean_std_dict = defaultdict(dict)
    for emotion_idx in df_['emotion'].unique():
        # emotion = i
        means = df_[df_['emotion'] == emotion_idx][choice_col].mean(axis=0).values
        stds = df_[df_['emotion'] == emotion_idx][choice_col].std(axis=0).values

        tmp_dict = {
            'valence_mean': means[0],
            'arousal_mean' : means[1],
            'valence_std' : stds[0],
            'arousal_std': stds[1]
        }
        mean_std_dict[emotion_r.get(emotion_idx)] = tmp_dict
    return mean_std_dict
