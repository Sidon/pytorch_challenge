import matplotlib.pyplot as plt
import numpy as np


def image_show(in_feature, title=None, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    """Imshow for Tensor."""
    in_feature = in_feature.numpy().transpose((1, 2, 0))
    mean = mean
    std = std
    in_feature = std * in_feature + mean
    in_feature = np.clip(in_feature, 0, 1)
    plt.imshow(in_feature)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    
    