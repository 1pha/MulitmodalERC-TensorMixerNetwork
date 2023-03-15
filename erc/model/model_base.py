""" Base for training
"""
import torch
from transformers import Wav2Vec2ForSequenceClassification, BertForSequenceClassification

from erc.constants import Task

class ModelBase:
    def __init__(self,):
        pass