import numpy as np
import torch
from scipy.stats import spearmanr as spr
from skimage.metrics import structural_similarity as ssim
from skimage import feature

# Sanity Checks for Saliency Methods
# https://arxiv.org/pdf/1810.03292.pdf

def get_layers(model):
    layers = []
    for name, _ in model.named_parameters():
        layer = name.split(".")[0]
        # Found a new layer
        if layer not in layers:
            layers.append(layer)

    return layers

# independent randomization code
# change layer num index to random values
def independent_layer_rand(model, empty_model, layer):   
    params = model.state_dict()
    params_new = empty_model.state_dict()

    # Set layer [index] to default params
    for name, _ in model.named_parameters():
        if name.startswith(layer):
            # if empty_model weights are all 0 for the parameter
            if torch.nonzero(params_new[name]).numel() == 0:
                params[name] = torch.rand(params[name].shape)
            else:
                params[name] = params_new[name]

    # load the modified paramaters into the model
    model.load_state_dict(params)
        
    return model

# cascading randomization code
# change layers 0 -> num index to random values
def cascading_layer_rand(model, empty_model, layers, index):
    params = model.state_dict()
    params_new = empty_model.state_dict()

    for i in range(index + 1):
        for name, _ in model.named_parameters():
            if name.startswith(layers[i]):
                # if empty_model weights are all 0 for the parameter
                if torch.nonzero(params_new[name]).numel() == 0:
                    params[name] = torch.rand(params[name].shape)
                else:
                    params[name] = params_new[name]

    # load the modified paramaters into the model
    model.load_state_dict(params)

    return model
   
def normalize_image(x):
    x = np.array(x).astype(np.float32)

    # fix any outliers
    x[x == float("Inf")] = 0
    x[x == float("-Inf")] = 0

    # normalize
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm

# evaluate the attr for normal network against attr for randomized network
def evaluate(normal_attr, random_layer_attr, abs = False):
    normal_attr_0_1 = normalize_image(normal_attr)
    random_layer_attr_0_1 = normalize_image(random_layer_attr)

    spr_val, _ = spr(normal_attr.flatten(), random_layer_attr.flatten())
    
    if abs == False:
        ssim_val = ssim(normal_attr_0_1, random_layer_attr_0_1, gaussian_weights = True, channel_axis = 2)
        normal_hog = feature.hog(normal_attr_0_1, pixels_per_cell = (16, 16), channel_axis = 2)
        rand_hog = feature.hog(random_layer_attr_0_1, pixels_per_cell = (16, 16), channel_axis = 2)
    else:
        ssim_val = ssim(normal_attr_0_1, random_layer_attr_0_1, gaussian_weights = True)
        normal_hog = feature.hog(normal_attr_0_1, pixels_per_cell = (16, 16))
        rand_hog = feature.hog(random_layer_attr_0_1, pixels_per_cell = (16, 16))

    hog_val, _ = spr(normal_hog, rand_hog)

    return ssim_val, spr_val, hog_val