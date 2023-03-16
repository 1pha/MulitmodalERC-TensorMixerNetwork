""" Base for training
"""
from torch import nn


class ModelBase(nn.Module):
    def __init__(self,):
        super().__init__()