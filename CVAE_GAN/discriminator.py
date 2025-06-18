import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
from PIL import Image
import os
import wandb
import clip
from tqdm import tqdm
import multiprocessing as mp
import typer


class Discriminator(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        ...
