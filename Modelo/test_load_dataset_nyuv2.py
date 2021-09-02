#!/usr/bin/env python3

from pathlib import Path
from nyuv2 import *
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = Path('dataset')

def plot_color(ax, color, title="Color"):
    """Displays a color image from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(color)

def plot_depth(ax, depth, title="Depth"):
    """Displays a depth map from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(depth, cmap='Spectral')

def plot_label(ax, labels, title="Label"):
    """Displays a label map from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(labels)

def test_labeled_dataset():
    labeled = LabeledDataset(DATASET_DIR / 'nyu_depth_v2_labeled.mat')

    color, depth, label = labeled[15]

    fig = plt.figure("Labeled Dataset Sample", figsize=(12, 5))

    ax = fig.add_subplot(1, 3, 1)
    plot_color(ax, color)

    ax = fig.add_subplot(1, 3, 2)
    plot_depth(ax, np.asarray(depth))

    ax = fig.add_subplot(1, 3, 3)
    plot_label(ax, np.asarray(label))

    plt.show()

    labeled.close()

test_labeled_dataset()
