import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision import transforms
import os 
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize as skimage_resize
from util.visualization import attr_to_subplot
from util.test_methods import MASTestFunctions as MAS_functions
from matplotlib.colors import LinearSegmentedColormap

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

cmap = LinearSegmentedColormap.from_list('custom blue',  [(0, (1, 1, 1, 1)), (0.25, '#0000ff'), (1, '#0000ff')], N = 256)   
# heatmap
def heatmap_overlap(img, attr, name):
    fig, axs = plt.subplots(1, 1)
    # attr_to_subplot(attr, name, axs, cmap = 'jet', norm = 'absolute', blended_image = img, alpha = 0.5)
    # attr_to_subplot(attr, name, axs, norm = 'absolute', blended_image = img, alpha = 0.9)
    attr_to_subplot(attr, name, axs, cmap = cmap, norm = 'absolute', blended_image = img, alpha = 0.7)

    return

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""

    # create nxn zeros
    inp = np.zeros((klen, klen))

    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1

    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k

    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

def normalize_curve(normalized_model_response, type = 0):
    n = len(normalized_model_response)
    Q = matrix(2 * np.eye(n))
    c = matrix(-2 * normalized_model_response)
    A_ineq = np.zeros((n - 2, n))
    b_ineq = np.full(n - 2, 0)
    row_indices = np.arange(n - 2)
    if type == 0:
        A_ineq[row_indices, row_indices] = -1
        A_ineq[row_indices, row_indices + 1] = 2
        A_ineq[row_indices, row_indices + 2] = -1
    elif type == 1:
        A_ineq[row_indices, row_indices] = 1
        A_ineq[row_indices, row_indices + 1] = -2
        A_ineq[row_indices, row_indices + 2] = 1
    G_bounds = np.vstack([-np.eye(n), np.eye(n)])
    h_bounds = np.hstack([np.zeros(n), np.ones(n)])
    G = matrix(np.vstack([G_bounds, A_ineq]))
    h = matrix(np.hstack([h_bounds, b_ineq]))
    A_eq = np.zeros((2, n))
    A_eq[0, 0] = 1
    A_eq[1, -1] = 1
    A_eq = matrix(A_eq)
    b_eq = matrix(np.array([normalized_model_response[0], normalized_model_response[-1]]), (2, 1), 'd')
    sol = solvers.qp(Q, c, G, h, A_eq, b_eq)
    return np.squeeze(np.array(sol['x']))

def find_best_segment_reverse_informed(input_tensor, saliency_map_segmented, segments, blur, n_searches, type, model, device, img_hw, max_batch_size = 25, cutoff = 0.9):
    if cutoff == 0:
        return 0, 0, torch.tensor([]), torch.tensor([0])

    n_steps = len(np.unique(segments))
    
    batch_size = n_steps if n_steps < 50 else 25
    batch_size = max_batch_size if batch_size > max_batch_size else batch_size
    if batch_size > n_steps:
        print("Batch size cannot be greater than number of steps: " + str(n_steps))
        return 0

    # Retrieve softmax score of the original image
    original_pred = model(input_tensor.to(device)).detach()
    _, index = torch.max(original_pred, 1)
    target_class = index[0]
    percentage = torch.nn.functional.softmax(original_pred, dim = 1)[0]
    original_pred = percentage[target_class].item()

    # deletion
    if type == 0:
        # set the start and stop images for each test
        start = torch.zeros_like(input_tensor)
        finish = input_tensor.clone()

        black_pred = model(start.to(device)).detach()
        percentage = torch.nn.functional.softmax(black_pred, dim = 1)[0]
        black_pred = percentage[target_class].item()
    # insertion
    elif type == 1:
        # set the start and stop images for each test
        start = blur(input_tensor)
        finish = input_tensor.clone()

        blur_pred = model(start.to(device)).detach()
        percentage = torch.nn.functional.softmax(blur_pred, dim = 1)[0]
        blur_pred = percentage[target_class].item()

    saliency_map = torch.ones((224, 224, 3))
    saliency_map_test = torch.abs(torch.sum(saliency_map.squeeze(), axis = 2))

    saliency_map_segmented_test = torch.abs(torch.sum(saliency_map_segmented.squeeze(), axis = 2))
    segment_saliency = torch.zeros(n_steps)
    for i in range(n_steps):
        segment = torch.where(segments.flatten() == i)[0]
        segment_saliency[i] = torch.mean(saliency_map_segmented_test.reshape(img_hw ** 2)[segment])
    
    if type == 0:
        # segments ordered from lowest to highest attr value
        segment_order = torch.argsort(segment_saliency)
    elif type == 1:
        # segments ordered from highest to lowest attr value
        segment_order = torch.flip(torch.argsort(segment_saliency), dims=(0,))

    num_batches = int((n_steps) / batch_size)
    leftover = (n_steps) % batch_size
    batches = torch.full((1, num_batches + 1), batch_size).squeeze()
    batches[-1] = leftover
    worst_segment_list = torch.full((1, n_steps), -1).squeeze()
    worst_MR_list = torch.empty((1, n_steps)).squeeze()

    subsearch_len = (int(n_steps ** (1/2)) * 2) if (int(n_steps ** (1/2)) * 2) <= 28 else 28
    for step in range(n_searches - subsearch_len):
        model_response = torch.zeros(subsearch_len)
        # find the lowest subsearch_len segments that have not already been selected
        selected_segments = torch.zeros(subsearch_len)
        search_counter = 0
        index_counter = 0
        while index_counter != subsearch_len:
            if segment_order[search_counter] not in worst_segment_list:
                selected_segments[index_counter] = segment_order[search_counter]
                index_counter += 1
                search_counter += 1 
            else:
                search_counter += 1 

        batch_size = subsearch_len if subsearch_len < batch_size else batch_size 
        num_batches = int(subsearch_len / batch_size)
        leftover = subsearch_len % batch_size
        batches = torch.full((1, num_batches + 1), batch_size).squeeze()
        batches[-1] = leftover
        total_steps = 0

        for batch in batches:
            images = torch.zeros((batch.item(), input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                start_temp = start.clone()
                segment_coords = torch.where(segments.flatten() == selected_segments[total_steps])[0]
                start_temp.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                images[i] = start_temp
                total_steps += 1
            
            # get predictions from image batch
            output = model(images.to(device)).detach()
            percentage = nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class]

        if type == 0:
            worst_MR_index = torch.argmin(model_response)
        elif type == 1:
            worst_MR_index = torch.argmax(model_response)

        worst_MR = model_response[worst_MR_index]
        worst_MR_list[step] = worst_MR
        worst_segment = selected_segments[worst_MR_index]
        worst_segment_list[step] = worst_segment

        segment_coords = torch.where(segments.flatten() == worst_segment)[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

        if cutoff != 1 and ((worst_MR - blur_pred) / (abs(original_pred - blur_pred))) >= cutoff:
            worst_MR_list[step] = cutoff
            return 0, 0, worst_segment_list, worst_MR_list

    subsearch_len_orig = subsearch_len
    for step in range(subsearch_len_orig):
        subsearch_len = subsearch_len_orig - step
        model_response = torch.zeros(subsearch_len)
        # find the lowest subsearch_len segments that have not already been selected
        selected_segments = torch.zeros(subsearch_len)
        search_counter = 0
        index_counter = 0
        while index_counter != subsearch_len:
            if segment_order[search_counter] not in worst_segment_list:
                selected_segments[index_counter] = segment_order[search_counter]
                index_counter += 1
                search_counter += 1 
            else:
                search_counter += 1 

        batch_size = subsearch_len if subsearch_len < batch_size else batch_size 
        num_batches = int(subsearch_len / batch_size)
        leftover = subsearch_len % batch_size
        batches = torch.full((1, num_batches + 1), batch_size).squeeze()
        batches[-1] = leftover
        total_steps = 0
        for batch in batches:
            images = torch.zeros((batch.item(), input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                start_temp = start.clone()
                segment_coords = torch.where(segments.flatten() == selected_segments[total_steps])[0]
                start_temp.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                images[i] = start_temp

                total_steps += 1

            # get predictions from image batch
            output = model(images.to(device)).detach()
            percentage = nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class]

        if type == 0:
            worst_MR_index = torch.argmin(model_response)
        elif type == 1:
            worst_MR_index = torch.argmax(model_response)

        worst_MR = model_response[worst_MR_index]
        worst_MR_list[step + n_searches - subsearch_len_orig] = worst_MR
        worst_segment = selected_segments[worst_MR_index]
        worst_segment_list[step + n_searches - subsearch_len_orig] = worst_segment

        segment_coords = torch.where(segments.flatten() == worst_segment)[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

        if cutoff != 1 and ((worst_MR - blur_pred) / (abs(original_pred - blur_pred))) >= cutoff:
            worst_MR_list[step] = cutoff
            return 0, 0, worst_segment_list, worst_MR_list

    if type == 0:
        normalized_model_response = torch.cat((worst_MR_list, torch.tensor([original_pred])))
        normalized_model_response = torch.flip(normalized_model_response, (0,))
    elif type == 1:
        normalized_model_response = torch.cat((torch.tensor([blur_pred]), worst_MR_list))

    min_normalized_pred = 1.0
    max_normalized_pred = 0.0
    # perform monotonic normalization of raw model response
    normalized_model_response = normalized_model_response.detach().cpu().numpy().copy().astype(np.double)

    for i in range(n_steps + 1):           
        if type == 0:
            normalized_pred = (normalized_model_response[i] - black_pred) / (abs(original_pred - black_pred))
            normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
            min_normalized_pred = min(min_normalized_pred, normalized_pred)
            normalized_model_response[i] = min_normalized_pred
        elif type == 1:
            normalized_pred = (normalized_model_response[i] - blur_pred) / (abs(original_pred - blur_pred))
            normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
            max_normalized_pred = max(max_normalized_pred, normalized_pred)
            normalized_model_response[i] = max_normalized_pred

    original_MR = normalized_model_response.copy()
    normalized_model_response = normalize_curve(normalized_model_response, type)

    if type == 0:
        best_segment_list = torch.flip(worst_segment_list, (0,))
    elif type == 1:
        best_segment_list = worst_segment_list

    new_map = torch.zeros_like(saliency_map_test)
    ones_map = torch.ones_like(saliency_map_test)
    for i in range(1, n_steps + 1):
        segment_coords = torch.where(segments.flatten() == best_segment_list[i-1])[0]
        if type == 0:
            target_MR = normalized_model_response[i - 1] - normalized_model_response[i]
        elif type == 1:
            target_MR = normalized_model_response[i] - normalized_model_response[i - 1]

        new_map.reshape(img_hw ** 2)[segment_coords] = (ones_map.reshape(img_hw ** 2)[segment_coords] / len(segment_coords)) * target_MR

    new_map = new_map.unsqueeze(2) * torch.ones_like(saliency_map)
    upsize = transforms.Resize(img_hw, antialias=True)
    small_side = int(np.ceil(np.sqrt(n_steps)))
    downsize = transforms.Resize(small_side, antialias=True)

    return new_map.cpu().numpy(), upsize(downsize(new_map.permute((2, 0, 1)))).permute((1, 2, 0)).cpu().numpy(), best_segment_list, original_MR

def find_deletion_from_insertion_informed(input_tensor, segments, saliency_map_segmented, beginning_order, n_searches, model, device, img_hw, max_batch_size = 25, gamma = 0):
    n_steps = len(np.unique(segments))
    
    batch_size = n_steps if n_steps < 50 else 25
    batch_size = max_batch_size if batch_size > max_batch_size else batch_size
    if batch_size > n_steps:
        print("Batch size cannot be greater than number of steps: " + str(n_steps))
        return 0

    # Retrieve softmax score of the original image
    original_pred = model(input_tensor.to(device)).detach()
    _, index = torch.max(original_pred, 1)
    target_class = index[0]
    percentage = torch.nn.functional.softmax(original_pred, dim = 1)[0]
    original_pred = percentage[target_class].item()

    klen = 31
    ksig = 31
    kern = gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)

    start = torch.zeros_like(input_tensor)
    
    finish = input_tensor.clone()
    black_pred = model(start.to(device)).detach()
    percentage = torch.nn.functional.softmax(black_pred, dim = 1)[0]
    black_pred = percentage[target_class].item()

    saliency_map = torch.ones((224, 224, 3))
    saliency_map_test = torch.abs(torch.sum(saliency_map.squeeze(), axis = 2))

    saliency_map_segmented_test = torch.abs(torch.sum(saliency_map_segmented.squeeze(), axis = 2))
    segment_saliency = torch.zeros(n_steps)
    for i in range(n_steps):
        segment = torch.where(segments.flatten() == i)[0]
        segment_saliency[i] = torch.mean(saliency_map_segmented_test.reshape(img_hw ** 2)[segment])
    
    # We want to find the worst possible insertion order, so order segments from least to most attr value
    segment_order = torch.argsort(segment_saliency)

    worst_segment_list = torch.full((1, n_steps), -1).squeeze()
    worst_MR_list = torch.zeros((1, n_steps)).squeeze()

    # Set the last elements of the segment list to be the input from insertion since we know those are the best insertion values
    input_length = len(beginning_order)
    worst_segment_list[len(worst_segment_list) - input_length : len(worst_segment_list)] = torch.flip(beginning_order, (0,))
    
    subsearch_len = (int(n_steps ** (1/2)) * 2) if (int(n_steps ** (1/2)) * 2) <= 28 else 28
    for step in range(n_searches - subsearch_len - input_length):
        model_response = torch.zeros(subsearch_len)
        
        # find the lowest subsearch_len segments that have not already been selected
        selected_segments = torch.zeros(subsearch_len)
        search_counter = 0
        index_counter = 0
        while index_counter != subsearch_len:
            if segment_order[search_counter] not in worst_segment_list:
                selected_segments[index_counter] = segment_order[search_counter]
                index_counter += 1
                search_counter += 1 
            else:
                search_counter += 1 

        batch_size = subsearch_len if subsearch_len < batch_size else batch_size 
        num_batches = int(subsearch_len / batch_size)
        leftover = subsearch_len % batch_size
        batches = torch.full((1, num_batches + 1), batch_size).squeeze()
        batches[-1] = leftover
        total_steps = 0
        for batch in batches:
            images = torch.zeros((batch.item(), input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                start_temp = start.clone()
                segment_coords = torch.where(segments.flatten() == selected_segments[total_steps])[0]
                start_temp.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                images[i] = start_temp

                total_steps += 1

            # get predictions from image batch
            output = model(images.to(device)).detach()
            percentage = nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class]

        worst_MR_index = torch.argmin(model_response)
        worst_MR = model_response[worst_MR_index]
        worst_MR_list[step] = worst_MR
        worst_segment = selected_segments[worst_MR_index]
        worst_segment_list[step] = worst_segment

        segment_coords = torch.where(segments.flatten() == worst_segment)[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

    if input_length > n_searches - subsearch_len:
        subsearch_len_orig = n_searches - input_length
    else:
        subsearch_len_orig = subsearch_len

    for step in range(subsearch_len_orig):
        subsearch_len = subsearch_len_orig - step

        model_response = torch.zeros(subsearch_len)
        # find the lowest subsearch_len segments that have not already been selected
        selected_segments = torch.zeros(subsearch_len)
        search_counter = 0
        index_counter = 0

        while index_counter != subsearch_len:
            if segment_order[search_counter] not in worst_segment_list:
                selected_segments[index_counter] = segment_order[search_counter]
                index_counter += 1
                search_counter += 1 
            else:
                search_counter += 1 

        batch_size = subsearch_len if subsearch_len < batch_size else batch_size 
        num_batches = int(subsearch_len / batch_size)
        leftover = subsearch_len % batch_size
        batches = torch.full((1, num_batches + 1), batch_size).squeeze()
        batches[-1] = leftover
        total_steps = 0
        for batch in batches:
            images = torch.zeros((batch.item(), input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                start_temp = start.clone()
                segment_coords = torch.where(segments.flatten() == selected_segments[total_steps])[0]
                start_temp.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                images[i] = start_temp

                total_steps += 1

            # get predictions from image batch
            output = model(images.to(device)).detach()
            percentage = nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class]

        worst_MR_index = torch.argmin(model_response)
        worst_MR = model_response[worst_MR_index]
        worst_MR_list[step + n_searches - subsearch_len_orig - input_length] = worst_MR
        worst_segment = selected_segments[worst_MR_index]
        worst_segment_list[step + n_searches - subsearch_len_orig - input_length] = worst_segment

        segment_coords = torch.where(segments.flatten() == worst_segment)[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

    worst_segment_list_length = len(worst_segment_list)
    for step in range(worst_segment_list_length - input_length , worst_segment_list_length):
        segment_coords = torch.where(segments.flatten() == worst_segment_list[step])[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

        # get predictions from image batch
        output = model(start.to(device)).detach()
        percentage = nn.functional.softmax(output, dim = 1)
        model_response = percentage[:, target_class]
        worst_MR_list[step] = model_response

    # this is the worst possible insertion curve
    normalized_model_response = torch.cat((worst_MR_list, torch.tensor([original_pred]))).detach().cpu().numpy().copy().astype(np.double)
    # flip it so it becomes the best possible deletion curve
    normalized_model_response = normalized_model_response[::-1]

    min_normalized_pred = 1.0
    # perform monotonic normalization of raw deletion model response
    for i in range(n_steps + 1):         
        normalized_pred = (normalized_model_response[i] - black_pred) / (abs(original_pred - black_pred))
        normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
        min_normalized_pred = min(min_normalized_pred, normalized_pred)
        normalized_model_response[i] = min_normalized_pred

    # fix the derivative of the deletion curve
    normalized_model_response = normalize_curve(normalized_model_response, type = 0)

    # flip the worst ordered insertion list to become the best ordered deletion list 
    best_segment_list = torch.flip(worst_segment_list, (0,))
    # using the model response, set each patch to be the correct value for perfect order deletion
    new_map = torch.zeros_like(saliency_map_test)
    ones_map = torch.ones_like(saliency_map_test)
    for i in range(1, n_steps + 1):
        segment_coords = torch.where(segments.flatten() == best_segment_list[i-1])[0]
        target_MR = normalized_model_response[i - 1] - normalized_model_response[i]
        new_map.reshape(img_hw ** 2)[segment_coords] = (ones_map.reshape(img_hw ** 2)[segment_coords] / len(segment_coords)) * target_MR + (target_MR * (n_steps - i) / n_steps)
    new_map = new_map.unsqueeze(2) * torch.ones_like(saliency_map)

    # get insertion and deletion curves for the current attribution
    MAS_insertion = MAS_functions.MASMetric(model, img_hw ** 2, 'ins', img_hw, substrate_fn = blur)
    MAS_deletion = MAS_functions.MASMetric(model, img_hw ** 2, 'del', img_hw, substrate_fn = torch.zeros_like)
    new_map_test = np.abs(np.sum(new_map.cpu().numpy(), axis = 2))
    _, _, _, _, raw_score_ins = MAS_insertion.single_run(input_tensor, new_map_test, device, max_batch_size=5)
    _, _, _, _, raw_score_del = MAS_deletion.single_run(input_tensor, new_map_test, device, max_batch_size=5)
    # interpolate the two curves to the correct number of steps
    x_old = np.linspace(0, 100, len(raw_score_ins))
    x_new = np.linspace(0, 100, n_steps + 1)
    raw_score_ins = np.interp(x_new, x_old, raw_score_ins)
    raw_score_del = np.interp(x_new, x_old, raw_score_del)
    # take the mean of the two curves, treat it as a deletion curve
    new_curve = np.mean([raw_score_ins, (raw_score_del * -1) + 1], axis = 0)
    new_curve = (new_curve - 1) * -1
    # fix derivative, this is the new model response
    normalized_model_response = normalize_curve(new_curve, type = 0)

    # using the model response, set each patch to be the correct value for perfect order deletion
    new_map_sparse = torch.zeros_like(saliency_map_test)
    new_map_dense = torch.zeros_like(saliency_map_test)
    ones_map = torch.ones_like(saliency_map_test)
    for i in range(1, n_steps + 1):
        segment_coords = torch.where(segments.flatten() == best_segment_list[i-1])[0]
        target_MR = normalized_model_response[i - 1] - normalized_model_response[i]
        attr_value = 1 / len(segment_coords) * target_MR + (target_MR * (n_steps - i) / n_steps) 

        new_map_sparse.reshape(img_hw ** 2)[segment_coords] = ones_map.reshape(img_hw ** 2)[segment_coords] * attr_value
        if attr_value >= 0.005:
            new_map_dense.reshape(img_hw ** 2)[segment_coords] = ones_map.reshape(img_hw ** 2)[segment_coords] * ((n_steps - i) / n_steps)
        else:
            new_map_dense.reshape(img_hw ** 2)[segment_coords] = ones_map.reshape(img_hw ** 2)[segment_coords] * 0
   
    # put the dense and sparse maps on the same range
    new_map_dense = new_map_dense / new_map_dense.max() * new_map_sparse.max()
    # combine maps with gamma
    new_map = ((1 - gamma) * new_map_sparse) + (gamma * new_map_dense)
    new_map = new_map.unsqueeze(2) * torch.ones_like(saliency_map)

    upsize = transforms.Resize(img_hw, antialias=True)
    small_side = int(np.ceil(np.sqrt(n_steps)))
    downsize = transforms.Resize(small_side, antialias=True)

    return new_map.cpu().numpy(), upsize(downsize(new_map.permute((2, 0, 1)))).permute((1, 2, 0)).cpu().numpy(), best_segment_list

def find_deletion_from_insertion_informed_fast(input_tensor, segments, saliency_map_segmented, beginning_order, blur, n_searches, model, device, img_hw, max_batch_size = 25, jump_size = 14, mod = False):
    n_steps = len(np.unique(segments))
    
    batch_size = n_steps if n_steps < 50 else 25
    batch_size = max_batch_size if batch_size > max_batch_size else batch_size
    if batch_size > n_steps:
        print("Batch size cannot be greater than number of steps: " + str(n_steps))
        return 0

    # Retrieve softmax score of the original image
    original_pred = model(input_tensor.to(device)).detach()
    _, index = torch.max(original_pred, 1)
    target_class = index[0]
    percentage = torch.nn.functional.softmax(original_pred, dim = 1)[0]
    original_pred = percentage[target_class].item()

    start = torch.zeros_like(input_tensor)
    
    finish = input_tensor.clone()
    black_pred = model(start.to(device)).detach()
    percentage = torch.nn.functional.softmax(black_pred, dim = 1)[0]
    black_pred = percentage[target_class].item()

    saliency_map = torch.ones((224, 224, 3))
    saliency_map_test = torch.abs(torch.sum(saliency_map.squeeze(), axis = 2))

    saliency_map_segmented_test = torch.abs(torch.sum(saliency_map_segmented.squeeze(), axis = 2))
    segment_saliency = torch.zeros(n_steps)
    for i in range(n_steps):
        segment = torch.where(segments.flatten() == i)[0]
        segment_saliency[i] = torch.mean(saliency_map_segmented_test.reshape(img_hw ** 2)[segment])
    
    # We want to find the worst possible insertion order, so order segments from least to most attr value
    segment_order = torch.argsort(segment_saliency)

    worst_segment_list = torch.full((1, n_steps), -1).squeeze()
    worst_MR_list = torch.zeros((1, n_steps)).squeeze()

    # Set the last elements of the segment list to be the input from insertion since we know those are the best insertion values
    input_length = len(beginning_order)
    worst_segment_list[len(worst_segment_list) - input_length : len(worst_segment_list)] = torch.flip(beginning_order, (0,))
    
    subsearch_len = (int(n_steps ** (1/2)) * 2) if (int(n_steps ** (1/2)) * 2) <= 28 else 28
    for step in range(0, n_searches - subsearch_len - input_length, jump_size):
        model_response = torch.zeros(subsearch_len)
        
        # find the lowest subsearch_len segments that have not already been selected
        selected_segments = torch.zeros(subsearch_len)
        search_counter = 0
        index_counter = 0
        while index_counter != subsearch_len:
            if segment_order[search_counter] not in worst_segment_list:
                selected_segments[index_counter] = segment_order[search_counter]
                index_counter += 1
                search_counter += 1 
            else:
                search_counter += 1 

        batch_size = subsearch_len if subsearch_len < batch_size else batch_size 
        num_batches = int(subsearch_len / batch_size)
        leftover = subsearch_len % batch_size
        batches = torch.full((1, num_batches + 1), batch_size).squeeze()
        batches[-1] = leftover
        total_steps = 0
        for batch in batches:
            images = torch.zeros((batch.item(), input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                start_temp = start.clone()
                segment_coords = torch.where(segments.flatten() == selected_segments[total_steps])[0]
                start_temp.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                images[i] = start_temp

                total_steps += 1

            # get predictions from image batch
            output = model(images.to(device)).detach()
            percentage = nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class]

        worst_MR_indices = torch.argsort(model_response)[0:jump_size]
        worst_MR_list[step : step + jump_size] = model_response[worst_MR_indices]
        worst_segment_list[step : step + jump_size] = selected_segments[worst_MR_indices]

        for i in range(jump_size):
            segment_coords = torch.where(segments.flatten() == worst_segment_list[step + i])[0]
            start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

    if input_length > n_searches - subsearch_len:
        subsearch_len_orig = n_searches - input_length
    else:
        subsearch_len_orig = subsearch_len

    subsearch_len_orig = len(torch.where(worst_segment_list == -1)[0])
    for step in range(subsearch_len_orig):
        subsearch_len = subsearch_len_orig - step

        model_response = torch.zeros(subsearch_len)
        # find the lowest subsearch_len segments that have not already been selected
        selected_segments = torch.zeros(subsearch_len)
        search_counter = 0
        index_counter = 0

        while index_counter != subsearch_len:
            if segment_order[search_counter] not in worst_segment_list:
                selected_segments[index_counter] = segment_order[search_counter]
                index_counter += 1
                search_counter += 1 
            else:
                search_counter += 1 

        batch_size = subsearch_len if subsearch_len < batch_size else batch_size 
        num_batches = int(subsearch_len / batch_size)
        leftover = subsearch_len % batch_size
        batches = torch.full((1, num_batches + 1), batch_size).squeeze()
        batches[-1] = leftover
        total_steps = 0
        for batch in batches:
            images = torch.zeros((batch.item(), input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                start_temp = start.clone()
                segment_coords = torch.where(segments.flatten() == selected_segments[total_steps])[0]
                start_temp.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                images[i] = start_temp

                total_steps += 1

            # get predictions from image batch
            output = model(images.to(device)).detach()
            percentage = nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class]

        worst_MR_index = torch.argmin(model_response)
        worst_MR = model_response[worst_MR_index]
        worst_MR_list[step + n_searches - subsearch_len_orig - input_length] = worst_MR
        worst_segment = selected_segments[worst_MR_index]
        worst_segment_list[step + n_searches - subsearch_len_orig - input_length] = worst_segment

        segment_coords = torch.where(segments.flatten() == worst_segment)[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

    worst_segment_list_length = len(worst_segment_list)
    for step in range(worst_segment_list_length - input_length , worst_segment_list_length):
        segment_coords = torch.where(segments.flatten() == worst_segment_list[step])[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

        # get predictions from image batch
        output = model(start.to(device)).detach()
        percentage = nn.functional.softmax(output, dim = 1)
        model_response = percentage[:, target_class]
        worst_MR_list[step] = model_response

    # this is the worst possible insertion curve
    normalized_model_response = torch.cat((worst_MR_list, torch.tensor([original_pred]))).detach().cpu().numpy().copy().astype(np.double)
    # flip it so it becomes the best possible deletion curve
    normalized_model_response = normalized_model_response[::-1]

    min_normalized_pred = 1.0
    # perform monotonic normalization of raw deletion model response
    for i in range(n_steps + 1):         
        normalized_pred = (normalized_model_response[i] - black_pred) / (abs(original_pred - black_pred))
        normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
        min_normalized_pred = min(min_normalized_pred, normalized_pred)
        normalized_model_response[i] = min_normalized_pred

    # fix the derivative of the deletion curve
    normalized_model_response = normalize_curve(normalized_model_response, type = 0)

    # flip the worst ordered insertion list to become the best ordered deletion list 
    best_segment_list = torch.flip(worst_segment_list, (0,))
    # using the model response, set each patch to be the correct value for perfect order deletion
    new_map = torch.zeros_like(saliency_map_test)
    ones_map = torch.ones_like(saliency_map_test)
    for i in range(1, n_steps + 1):
        segment_coords = torch.where(segments.flatten() == best_segment_list[i-1])[0]
        target_MR = normalized_model_response[i - 1] - normalized_model_response[i]
        new_map.reshape(img_hw ** 2)[segment_coords] = (ones_map.reshape(img_hw ** 2)[segment_coords] / len(segment_coords)) * target_MR + (target_MR * (n_steps - i) / n_steps)
    new_map = new_map.unsqueeze(2) * torch.ones_like(saliency_map)

    # get insertion and deletion curves for the current attribution
    MAS_insertion = MAS_functions.MASMetric(model, img_hw ** 2, 'ins', img_hw, substrate_fn = blur)
    MAS_deletion = MAS_functions.MASMetric(model, img_hw ** 2, 'del', img_hw, substrate_fn = torch.zeros_like)
    new_map_test = np.abs(np.sum(new_map.cpu().numpy(), axis = 2))
    _, _, _, _, raw_score_ins = MAS_insertion.single_run(input_tensor, new_map_test, device, max_batch_size=5)
    _, _, _, _, raw_score_del = MAS_deletion.single_run(input_tensor, new_map_test, device, max_batch_size=5)
    # interpolate the two curves to the correct number of steps
    x_old = np.linspace(0, 100, len(raw_score_ins))
    x_new = np.linspace(0, 100, n_steps + 1)
    raw_score_ins = np.interp(x_new, x_old, raw_score_ins)
    raw_score_del = np.interp(x_new, x_old, raw_score_del)
    # take the mean of the two curves, treat it as a deletion curve
    new_curve = np.mean([raw_score_ins, (raw_score_del * -1) + 1], axis = 0)
    new_curve = (new_curve - 1) * -1
    # fix derivative, this is the new model response
    normalized_model_response = normalize_curve(new_curve, type = 0)

    # using the model response, set each patch to be the correct value for perfect order deletion
    new_map = torch.zeros_like(saliency_map_test)
    ones_map = torch.ones_like(saliency_map_test)
    for i in range(1, n_steps + 1):
        segment_coords = torch.where(segments.flatten() == best_segment_list[i-1])[0]
        target_MR = normalized_model_response[i - 1] - normalized_model_response[i]
        attr_value = 1 / len(segment_coords) * target_MR + (target_MR * (n_steps - i) / n_steps) 
        if mod == False:
            new_map.reshape(img_hw ** 2)[segment_coords] = ones_map.reshape(img_hw ** 2)[segment_coords] * attr_value
        elif mod == True:
            if attr_value >= 0.001:
                new_map.reshape(img_hw ** 2)[segment_coords] = ones_map.reshape(img_hw ** 2)[segment_coords] * ((n_steps - i) / n_steps) 
            else:
                new_map.reshape(img_hw ** 2)[segment_coords] = ones_map.reshape(img_hw ** 2)[segment_coords] * 0
    new_map = new_map.unsqueeze(2) * torch.ones_like(saliency_map)
    
    upsize = transforms.Resize(img_hw, antialias=True)
    small_side = int(np.ceil(np.sqrt(n_steps)))
    downsize = transforms.Resize(small_side, antialias=True)

    return new_map.cpu().numpy(), upsize(downsize(new_map.permute((2, 0, 1)))).permute((1, 2, 0)).cpu().numpy(), best_segment_list

def find_deletion_from_insertion_informed_ultimate(input_tensor, segments, saliency_map_segmented, beginning_order, blur, n_searches, model, device, img_hw, max_batch_size = 25, kappa = 0.005, test_kappa = False):
    n_steps = len(np.unique(segments))
    
    batch_size = n_steps if n_steps < 50 else 25
    batch_size = max_batch_size if batch_size > max_batch_size else batch_size
    if batch_size > n_steps:
        print("Batch size cannot be greater than number of steps: " + str(n_steps))
        return 0

    # Retrieve softmax score of the original image
    original_pred = model(input_tensor.to(device)).detach()
    _, index = torch.max(original_pred, 1)
    target_class = index[0]
    percentage = torch.nn.functional.softmax(original_pred, dim = 1)[0]
    original_pred = percentage[target_class].item()

    start = torch.zeros_like(input_tensor)
    
    finish = input_tensor.clone()
    black_pred = model(start.to(device)).detach()
    percentage = torch.nn.functional.softmax(black_pred, dim = 1)[0]
    black_pred = percentage[target_class].item()

    saliency_map = torch.ones((224, 224, 3))
    saliency_map_test = torch.abs(torch.sum(saliency_map.squeeze(), axis = 2))

    saliency_map_segmented_test = torch.abs(torch.sum(saliency_map_segmented.squeeze(), axis = 2))
    segment_saliency = torch.zeros(n_steps)
    for i in range(n_steps):
        segment = torch.where(segments.flatten() == i)[0]
        segment_saliency[i] = torch.mean(saliency_map_segmented_test.reshape(img_hw ** 2)[segment])
    
    # We want to find the worst possible insertion order, so order segments from least to most attr value
    segment_order = torch.argsort(segment_saliency)

    worst_segment_list = torch.full((1, n_steps), -1).squeeze()
    worst_MR_list = torch.zeros((1, n_steps)).squeeze()

    # Set the last elements of the segment list to be the input from insertion since we know those are the best insertion values
    input_length = len(beginning_order)
    worst_segment_list[len(worst_segment_list) - input_length : len(worst_segment_list)] = torch.flip(beginning_order, (0,))
    
    subsearch_len = (int(n_steps ** (1/2)) * 2) if (int(n_steps ** (1/2)) * 2) <= 28 else 28
    for step in range(n_searches - subsearch_len - input_length):
        model_response = torch.zeros(subsearch_len)
        
        # find the lowest subsearch_len segments that have not already been selected
        selected_segments = torch.zeros(subsearch_len)
        search_counter = 0
        index_counter = 0
        while index_counter != subsearch_len:
            if segment_order[search_counter] not in worst_segment_list:
                selected_segments[index_counter] = segment_order[search_counter]
                index_counter += 1
                search_counter += 1 
            else:
                search_counter += 1 

        batch_size = subsearch_len if subsearch_len < batch_size else batch_size 
        num_batches = int(subsearch_len / batch_size)
        leftover = subsearch_len % batch_size
        batches = torch.full((1, num_batches + 1), batch_size).squeeze()
        batches[-1] = leftover
        total_steps = 0
        for batch in batches:
            images = torch.zeros((batch.item(), input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                start_temp = start.clone()
                segment_coords = torch.where(segments.flatten() == selected_segments[total_steps])[0]
                start_temp.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                images[i] = start_temp

                total_steps += 1

            # get predictions from image batch
            output = model(images.to(device)).detach()
            percentage = nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class]

        worst_MR_index = torch.argmin(model_response)
        worst_MR = model_response[worst_MR_index]
        worst_MR_list[step] = worst_MR
        worst_segment = selected_segments[worst_MR_index]
        worst_segment_list[step] = worst_segment

        segment_coords = torch.where(segments.flatten() == worst_segment)[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

    if input_length > n_searches - subsearch_len:
        subsearch_len_orig = n_searches - input_length
    else:
        subsearch_len_orig = subsearch_len

    for step in range(subsearch_len_orig):
        subsearch_len = subsearch_len_orig - step

        model_response = torch.zeros(subsearch_len)
        # find the lowest subsearch_len segments that have not already been selected
        selected_segments = torch.zeros(subsearch_len)
        search_counter = 0
        index_counter = 0

        while index_counter != subsearch_len:
            if segment_order[search_counter] not in worst_segment_list:
                selected_segments[index_counter] = segment_order[search_counter]
                index_counter += 1
                search_counter += 1 
            else:
                search_counter += 1 

        batch_size = subsearch_len if subsearch_len < batch_size else batch_size 
        num_batches = int(subsearch_len / batch_size)
        leftover = subsearch_len % batch_size
        batches = torch.full((1, num_batches + 1), batch_size).squeeze()
        batches[-1] = leftover
        total_steps = 0
        for batch in batches:
            images = torch.zeros((batch.item(), input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                start_temp = start.clone()
                segment_coords = torch.where(segments.flatten() == selected_segments[total_steps])[0]
                start_temp.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                images[i] = start_temp

                total_steps += 1

            # get predictions from image batch
            output = model(images.to(device)).detach()
            percentage = nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class]

        worst_MR_index = torch.argmin(model_response)
        worst_MR = model_response[worst_MR_index]
        worst_MR_list[step + n_searches - subsearch_len_orig - input_length] = worst_MR
        worst_segment = selected_segments[worst_MR_index]
        worst_segment_list[step + n_searches - subsearch_len_orig - input_length] = worst_segment

        segment_coords = torch.where(segments.flatten() == worst_segment)[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

    worst_segment_list_length = len(worst_segment_list)
    for step in range(worst_segment_list_length - input_length , worst_segment_list_length):
        segment_coords = torch.where(segments.flatten() == worst_segment_list[step])[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

        # get predictions from image batch
        output = model(start.to(device)).detach()
        percentage = nn.functional.softmax(output, dim = 1)
        model_response = percentage[:, target_class]
        worst_MR_list[step] = model_response

    # this is the worst possible insertion curve
    normalized_model_response = torch.cat((worst_MR_list, torch.tensor([original_pred]))).detach().cpu().numpy().copy().astype(np.double)
    # flip it so it becomes the best possible deletion curve
    normalized_model_response = normalized_model_response[::-1]

    min_normalized_pred = 1.0
    # perform monotonic normalization of raw deletion model response
    for i in range(n_steps + 1):         
        normalized_pred = (normalized_model_response[i] - black_pred) / (abs(original_pred - black_pred))
        normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
        min_normalized_pred = min(min_normalized_pred, normalized_pred)
        normalized_model_response[i] = min_normalized_pred

    # fix the derivative of the deletion curve
    normalized_model_response = normalize_curve(normalized_model_response, type = 0)

    # flip the worst ordered insertion list to become the best ordered deletion list 
    best_segment_list = torch.flip(worst_segment_list, (0,))
    # using the model response, set each patch to be the correct value for perfect order deletion
    new_map = torch.zeros_like(saliency_map_test)
    ones_map = torch.ones_like(saliency_map_test)
    for i in range(1, n_steps + 1):
        segment_coords = torch.where(segments.flatten() == best_segment_list[i-1])[0]
        target_MR = normalized_model_response[i - 1] - normalized_model_response[i]
        new_map.reshape(img_hw ** 2)[segment_coords] = (ones_map.reshape(img_hw ** 2)[segment_coords] / len(segment_coords)) * target_MR + (target_MR * (n_steps - i) / n_steps)
    new_map = new_map.unsqueeze(2) * torch.ones_like(saliency_map)

    # get insertion and deletion curves for the current attribution
    MAS_insertion = MAS_functions.MASMetric(model, img_hw ** 2, 'ins', img_hw, substrate_fn = blur)
    MAS_deletion = MAS_functions.MASMetric(model, img_hw ** 2, 'del', img_hw, substrate_fn = torch.zeros_like)
    new_map_test = np.abs(np.sum(new_map.cpu().numpy(), axis = 2))
    _, _, _, _, raw_score_ins = MAS_insertion.single_run(input_tensor, new_map_test, device, max_batch_size=5)
    _, _, _, _, raw_score_del = MAS_deletion.single_run(input_tensor, new_map_test, device, max_batch_size=5)
    # interpolate the two curves to the correct number of steps
    x_old = np.linspace(0, 100, len(raw_score_ins))
    x_new = np.linspace(0, 100, n_steps + 1)
    raw_score_ins = np.interp(x_new, x_old, raw_score_ins)
    raw_score_del = np.interp(x_new, x_old, raw_score_del)
    # take the mean of the two curves, treat it as a deletion curve
    new_curve = np.mean([raw_score_ins, 1 - raw_score_del], axis = 0)
    new_curve = 1 - new_curve
    # fix derivative, this is the new model response
    normalized_model_response = normalize_curve(new_curve, type = 0)

    # using the model response, set each patch to be the correct value for perfect order deletion
    new_map_sparse = torch.zeros_like(saliency_map_test)
    new_map_dense = torch.zeros_like(saliency_map_test)
    ones_map = torch.ones_like(saliency_map_test)

    if test_kappa:
        return new_map_dense, segments, best_segment_list, normalized_model_response, ones_map, n_steps + 1

    for i in range(1, n_steps + 1):
        segment_coords = torch.where(segments.flatten() == best_segment_list[i-1])[0]
        target_MR = normalized_model_response[i - 1] - normalized_model_response[i]
        attr_value = ((1 / len(segment_coords)) * target_MR) + (target_MR * (n_steps - i) / n_steps) 

        # sparse attr, gamma = 0
        new_map_sparse.reshape(img_hw ** 2)[segment_coords] = ones_map.reshape(img_hw ** 2)[segment_coords] * attr_value

        # dense attr, gamma = 1
        if attr_value >= kappa:
            new_map_dense.reshape(img_hw ** 2)[segment_coords] = ones_map.reshape(img_hw ** 2)[segment_coords] * ((n_steps - i) / n_steps)
        else:
            new_map_dense.reshape(img_hw ** 2)[segment_coords] = ones_map.reshape(img_hw ** 2)[segment_coords] * attr_value

    # put the dense and sparse maps on the same range
    new_map_dense = new_map_dense / new_map_dense.max() * new_map_sparse.max()

    new_map_sparse = new_map_sparse
    new_map_5 = ((0.5) * new_map_sparse) + ((0.5) * new_map_dense)
    new_map_dense = new_map_dense
    new_map_sparse = new_map_sparse.unsqueeze(2) * torch.ones_like(saliency_map)
    new_map_5 = new_map_5.unsqueeze(2) * torch.ones_like(saliency_map)
    new_map_dense = new_map_dense.unsqueeze(2) * torch.ones_like(saliency_map)

    upsize = transforms.Resize(img_hw, antialias=True)
    small_side = int(np.ceil(np.sqrt(n_steps)))
    downsize = transforms.Resize(small_side, antialias=True)

    return new_map_sparse, new_map_dense

    # return new_map_0.cpu().numpy(), upsize(downsize(new_map_0.permute((2, 0, 1)))).permute((1, 2, 0)).cpu().numpy(), new_map_5.cpu().numpy(), upsize(downsize(new_map_5.permute((2, 0, 1)))).permute((1, 2, 0)).cpu().numpy(), new_map_10.cpu().numpy(), upsize(downsize(new_map_10.permute((2, 0, 1)))).permute((1, 2, 0)).cpu().numpy(), best_segment_list

def calibrate_density(input_tensor, saliency_map, model, device, total_steps, img_hw, type = 0, special_version = False):
    step_size = int(img_hw ** 2 / total_steps)
    n_steps = (img_hw ** 2 + step_size - 1) // step_size
    
    model_response = MAS(input_tensor, saliency_map, model, device, total_steps, img_hw, type = type, preprocess = 1, special_version = special_version)
    
    model_response = model_response.detach().cpu().numpy().copy().astype(np.double)
    n = len(model_response)

    # Objective function components
    Q = matrix(2 * np.eye(n))  # 2 * I because the objective function is (y - x)^2
    c = matrix(-2 * model_response)  # -2 * y for the objective function -2 * (y - x)

    # Derivative Constraints: 
    # x_2 - x_1 <= x_3 - x_2 (deletion)
    # x_2 - x_1 >= x_3 - x_2 (insertion)
    A_ineq = np.zeros((n - 2, n))
    b_ineq = np.full(n - 2, 0)
    row_indices = np.arange(n - 2)
    if type == 0:
        A_ineq[row_indices, row_indices] = -1
        A_ineq[row_indices, row_indices + 1] = 2
        A_ineq[row_indices, row_indices + 2] = -1
    elif type == 1:
        A_ineq[row_indices, row_indices] = 1
        A_ineq[row_indices, row_indices + 1] = -2
        A_ineq[row_indices, row_indices + 2] = 1

    # Bounds constraints: 0 <= x <= 1
    G_bounds = np.vstack([-np.eye(n), np.eye(n)])  # -x <= 0 and x <= 1
    h_bounds = np.hstack([np.zeros(n), np.ones(n)])  # Corresponds to bounds

    # Convert to cvxopt format
    G = matrix(np.vstack([G_bounds, A_ineq]))  # Combine G for bounds and A_ineq for monotonicity
    h = matrix(np.hstack([h_bounds, b_ineq]))  # Combine h for bounds and b_ineq for monotonicity

    # Endpoint constraints
    A_eq = np.zeros((2, n))
    A_eq[0, 0] = 1
    A_eq[1, -1] = 1
    A_eq = matrix(A_eq)
    b_eq = matrix(np.array([model_response[0], model_response[-1]]), (2, 1), 'd')

    # Solve and extract solution
    sol = solvers.qp(Q, c, G, h, A_eq, b_eq)
    model_response = np.squeeze(np.array(sol['x']))

    saliency_map_test = torch.abs(torch.sum(saliency_map.squeeze(), axis = 2))
    new_map = torch.zeros_like(saliency_map_test)
    ones_map = torch.ones_like(saliency_map_test)
    salient_order = torch.argsort(saliency_map_test.reshape(-1, img_hw ** 2), axis = 1).flip(-1)

    for i in range(1, n_steps+1):
        if type == 0:
            target_DR = model_response[i - 1] - model_response[i]
        elif type == 1:
            target_DR = model_response[i] - model_response[i - 1]

        coords = salient_order[:, step_size * (i - 1) : step_size * (i)]

        new_map.reshape(img_hw ** 2)[coords] = (ones_map.reshape(img_hw ** 2)[coords] / len(coords)) * target_DR

    new_map = new_map.unsqueeze(2) * torch.ones_like(saliency_map)

    return new_map

def remove_pixels(input_tensor, saliency_map, model, device, total_steps, img_hw, segments = None, type = 0, special_version = False):
    if segments is None:
        step_size = int(img_hw ** 2 / total_steps)
        n_steps = (img_hw ** 2 + step_size - 1) // step_size
    else:
        n_steps = len(np.unique(segments))
    
    model_response = MAS(input_tensor, saliency_map, model, device, total_steps, img_hw, segments = segments, type = type, preprocess = 1, special_version = special_version)
    # find derivative of the model response
    model_response_deriv = model_response.clone()
    if type == 0:
        model_response_deriv = np.insert(model_response_deriv.detach().numpy(), 0, 1)
    elif type == 1:
        model_response_deriv = np.insert(model_response_deriv.detach().numpy(), 1, 0)
    model_response_deriv = torch.abs(torch.diff(torch.tensor(model_response_deriv), 1))
    
    saliency_map_test = torch.abs(torch.sum(saliency_map.squeeze(), axis = 2))
    if segments is None:
        # Coordinates of pixels in order of decreasing saliency
        salient_order = torch.argsort(saliency_map_test.reshape(-1, img_hw ** 2), axis = 1).flip(-1)
    else:
        # segment numbers in order of decreasing saliency
        segment_saliency = torch.zeros(n_steps)
        for i in range(n_steps):
            segment = torch.where(segments.flatten() == i)[0]
            segment_saliency[i] = torch.mean(saliency_map_test.reshape(img_hw ** 2)[segment])
        salient_order = torch.flip(torch.argsort(segment_saliency), dims = (0, ))
        
    # remove pixels at end of curve where MR' = 0
    start = saliency_map_test.clone()
    finish = torch.zeros_like(saliency_map_test)
    # where the model response derivative has not yet become nonzero, penalty is 0
    step_to_start_removing = torch.where(model_response_deriv != 0)[0][-1]
    for i in range(step_to_start_removing, n_steps):
        if segments is None:
            coords = salient_order[:, step_size * (i - 1) : step_size * (i)]
        else:
            coords = torch.where(segments.flatten() == salient_order[i - 1])[0]

        start.reshape(1, img_hw ** 2)[:, coords] = finish.reshape(1, img_hw ** 2)[:, coords]
    preprocessed_attr = start.unsqueeze(2) * torch.ones_like(saliency_map)

    return preprocessed_attr

def MAS(input_tensor, saliency_map, model, device, total_steps, img_hw, segments = None, segment_order = None, type = 0, double_loss = 0, preprocess = 0, special_version = False):
    if segments is None:
        step_size = int(img_hw ** 2 / total_steps)
        n_steps = (img_hw ** 2 + step_size - 1) // step_size
    else:
        n_steps = len(np.unique(segments))

    batch_size = n_steps if n_steps < 50 else 50
    if batch_size > n_steps:
        print("Batch size cannot be greater than number of steps: " + str(n_steps))
        return 0

    # Retrieve softmax score of the original image
    original_pred = model(input_tensor.to(device)).detach()
    _, index = torch.max(original_pred, 1)
    target_class = index[0]
    percentage = nn.functional.softmax(original_pred, dim = 1)[0]
    original_pred = percentage[target_class]
    model_response = torch.zeros(n_steps + 1)

    # deletion
    if type == 0:
        start = input_tensor.clone()
        finish = torch.zeros_like(input_tensor, requires_grad = True)

        # # initialize ins blur kernel
        # klen = 31
        # ksig = 31
        # kern = gkern(klen, ksig)
        # blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)

        # # set the start and stop images for each test
        # # get softmax score of the subtrate-applied images
        # finish = blur(input_tensor)
        # # start = torch.zeros_like(input_tensor)
        # start = input_tensor.clone()

        black_pred = model(finish.to(device))
        percentage = nn.functional.softmax(black_pred, dim = 1)[0]
        black_pred = percentage[target_class]

        model_response[0] = original_pred
    # insertion
    elif type == 1:
        # initialize ins blur kernel
        klen = 31
        ksig = 31
        kern = gkern(klen, ksig)
        blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)

        # set the start and stop images for each test
        # get softmax score of the subtrate-applied images
        start = blur(input_tensor)
        # start = torch.zeros_like(input_tensor)
        finish = input_tensor.clone()

        blur_pred = model(start.to(device))
        percentage = nn.functional.softmax(blur_pred, dim = 1)[0]
        blur_pred = percentage[target_class]

        model_response[0] = blur_pred

    saliency_map_test = torch.abs(torch.sum(saliency_map.squeeze(), axis = 2))
    if segments is None:
        # Coordinates of pixels in order of decreasing saliency
        salient_order = torch.argsort(saliency_map_test.reshape(-1, img_hw ** 2), axis = 1).flip(-1)
    else:
        # segment numbers in order of decreasing saliency
        segment_saliency = torch.zeros(n_steps)
        for i in range(n_steps):
            segment = torch.where(segments.flatten() == i)[0]
            segment_saliency[i] = torch.mean(saliency_map_test.reshape(img_hw ** 2)[segment])

        if segment_order is None:
            segment_order = torch.flip(torch.argsort(segment_saliency), dims = (0, ))
        
    density_response = torch.zeros(n_steps + 1)
    if type == 0:
        density_response[0] = 1
    elif type == 1:
        density_response[0] = 0

    total_attr = torch.sum(saliency_map_test.reshape(1, 1, img_hw ** 2))
    step_num = 1
    num_batches = int((n_steps) / batch_size)
    leftover = (n_steps) % batch_size
    batches = torch.full((1, num_batches + 1), batch_size).squeeze()
    batches[-1] = leftover
    for batch in batches:
        images = torch.zeros((batch.item(), start.shape[1], start.shape[2], start.shape[3]))
        # collect images at all batch steps before mass prediction 
        for i in range(batch):
            if segments is None:
                coords = salient_order[:, step_size * (step_num - 1) : step_size * (step_num)]
                start.reshape(1, 3, img_hw ** 2)[0, :, coords] = finish.reshape(1, 3, img_hw ** 2)[0, :, coords]
                attr_count = torch.sum(saliency_map_test.reshape(1, 1, img_hw ** 2)[0, :, coords]) 
            else:
                segment_coords = torch.where(segments.flatten() == segment_order[step_num - 1])[0]
                start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                attr_count = torch.sum(saliency_map_test.reshape(img_hw ** 2)[segment_coords]) 

            images[i] = start

            if type == 0:
                density_response[step_num] = density_response[step_num - 1] - (attr_count / total_attr)
            elif type == 1:
                density_response[step_num] = density_response[step_num - 1] + (attr_count / total_attr)

            step_num += 1

        # get predictions from image batch
        output = model(images.to(device)).detach()
        percentage = nn.functional.softmax(output, dim = 1)
        model_response[step_num - batch : step_num] = percentage[:, target_class]

    # print(model_response)

    if special_version == True:
        model_response_deriv = torch.cat((torch.tensor([1.0]), model_response))
        model_response_deriv = torch.diff(model_response_deriv, 1)
        model_response_deriv[0] = model_response_deriv[1]

        # if we are performing deletion, we want the negative derivatives to be treated as positive for the next step
        if type == 0:
            model_response_deriv = model_response_deriv * -1

        # perform monotonicically decreasing normalization of MR deriv
        new_model_response_deriv = torch.zeros_like(model_response_deriv)
        min_normalized_pred = torch.tensor(1.0)
        for i in range(len(model_response_deriv)):   
            normalized_pred = (model_response_deriv[i] - model_response_deriv[-1]) / (model_response_deriv[0] - model_response_deriv[-1])

            if normalized_pred > 1:
                clipped_normalized_pred = min_normalized_pred
            elif normalized_pred < 0:
                clipped_normalized_pred = model_response_deriv[i - 1]
            else:
                clipped_normalized_pred = normalized_pred

            min_normalized_pred = torch.min(min_normalized_pred, clipped_normalized_pred)
            new_model_response_deriv[i] = min_normalized_pred

        model_response_deriv = new_model_response_deriv

        # undo the multiplication from earlier to obtain the correct derivative
        if type == 0:
            model_response_deriv = model_response_deriv * -1
 
        # print(model_response_deriv)

        # integrate the derivative to find the model response
        model_response = torch.cumsum(model_response_deriv, dim = 0)

        # print(model_response)

        model_response = (model_response - torch.min(model_response)) / (torch.max(model_response) - torch.min(model_response)) 
    else:
        # perform monotonic normalization of raw model response
        # normalized_model_response = model_response.clone()
        min_normalized_pred = 1.0
        max_normalized_pred = 0.0
        for i in range(n_steps + 1):           
            if type == 0:
                normalized_pred = (model_response[i] - black_pred) / (original_pred - black_pred)
                normalized_pred = torch.clip(normalized_pred, 0.0, 1.0)
                min_normalized_pred = min(min_normalized_pred, normalized_pred)
                model_response[i] = min_normalized_pred
            elif type == 1:
                normalized_pred = (model_response[i] - blur_pred) / (original_pred - blur_pred)
                normalized_pred = torch.clip(normalized_pred, 0.0, 1.0)
                max_normalized_pred = max(max_normalized_pred, normalized_pred)
                model_response[i] = max_normalized_pred

    # apply the alignment penalty to the model response
    alignment_penalty = torch.abs(model_response - density_response)
    if type == 0:
        corrected_scores = model_response + alignment_penalty      
    elif type == 1:
        corrected_scores = model_response - alignment_penalty

    # scores should be clipped before normalization or else values outside of these bounds will artificially improve the final score
    corrected_scores = corrected_scores.clip(0, 1)
    corrected_scores = (corrected_scores - corrected_scores.min()) / (corrected_scores.max() - corrected_scores.min())

    # if the blurred or black image recieved the same prediction as the input image, causing a failure of the above line, assign the attribution an ROC with a score of 0.5
    if torch.isnan(corrected_scores).any():
        if type == 0:
            corrected_scores = torch.linspace(1, 0, n_steps + 1, requires_grad = True)
        elif type == 1:
            corrected_scores = torch.linspace(0, 1, n_steps + 1, requires_grad = True)
    
    if preprocess == 1:
        return model_response

    # plt.rcParams.update({'font.size': 20})
    # plt.figure(figsize = (7, 4))
    # plt.title("MAS Ins*: " + str(round(auc(corrected_scores).item(), 3)))
    # plt.ylabel("Response")
    # plt.xlabel("Perturbation Percentage")
    # x = np.linspace(0, 100, n_steps + 1)
    # plt.plot(x, model_response.detach().numpy(), label = "Model Response", linewidth = 2)
    # plt.plot(x, density_response.detach().numpy(), label = "Density Response", linewidth = 2, color = "green")
    # plt.plot(x, alignment_penalty.detach().numpy(), label = "Alignment Penalty", linewidth = 2, color = "red")
    # plt.legend()
    # plt.show()

    if double_loss == 0:
        return auc(corrected_scores)
    else:
        return auc(corrected_scores), auc(alignment_penalty)

# A simple neural network with the attribution as the only trainable paramater
class Net(nn.Module):
    def __init__(self, attribution):
        super().__init__()
        self.original = attribution.clone()
        self.attribution = torch.nn.Parameter(data = attribution.clone().detach().requires_grad_(True))
    def forward(self):
        return self.attribution + self.original
    
class MASCalibrator():
    def __init__(self, model, img_hw, device):
        r""" Create an instance of the attribution calibrator.
        Args:
            model (nn.Module): Black-box model being explained.
            img_hw (int): The side length of the image in pixels, e.g. 224.
            device (str): Given as 'cpu' or gpu id e.g. 'cuda:0'.
        """
        self.model = model
        self.img_hw = img_hw
        self.device = device
    
        # set the parameters which perform the attribution smoothing
        self.upsize_a = transforms.Resize(img_hw, interpolation=transforms.InterpolationMode.NEAREST_EXACT, antialias=True)
        self.upsize_b = transforms.Resize(img_hw, antialias=True)
        small_side = int(np.ceil(np.sqrt(img_hw)))
        self.downsize = transforms.Resize(7, interpolation=transforms.InterpolationMode.NEAREST_EXACT, antialias=True)

    def refine_attribution(self, input_tensor, saliency_map, lr = None, smoothing = False, epochs = 25, segments = None, segment_order = None, special_version = False):
        r""" Return a refined version of the input saliency map with improved MAS scores.
        Args:
            input_tensor (Torch.tensor): Normalized input image in shape torch.Size([1, 3, 224, 224]).
            saliency_map (np.ndarray): The attribution to refine in shape (224, 224, 3).
            smoothing (bool): If True, apply a downscaling smoothing to force pixelated attributions into contiguous attributions.
            epochs (int): The number of optimization steps to perform.
            segments (np.ndarray) [optional]: A 224*224 integer array which indicates each of the N segment locations in the image starting from 0.
                                              If given, the perturbation order is determined by the mean attribution of each segment.
        """

        # convert the attribution to a torch tensor
        if smoothing == False:
            attr_to_clean = torch.tensor(saliency_map, dtype = torch.float).to(self.device)
        else:
            # downsize and then upsize the attribution, effectively smoothing a per-pixel attribution into contiguous regions
            attr_to_clean = self.upsize_b(self.downsize(torch.tensor(saliency_map, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0)).to(self.device)

        if segments is not None:
            attr_to_clean_test = torch.abs(torch.sum(attr_to_clean.squeeze(), axis = 2))
            # segment numbers in order of decreasing saliency
            segment_saliency = torch.zeros(len(torch.unique(segments)))
            for i in range(len(torch.unique(segments))):
                segment = torch.where(segments.flatten() == i)[0]
                segment_saliency[i] = torch.mean(attr_to_clean_test.reshape(self.img_hw ** 2)[segment])
                attr_to_clean_test.reshape(self.img_hw ** 2)[segment] = segment_saliency[i]

            attr_to_clean = attr_to_clean_test.unsqueeze(2) * torch.ones_like(attr_to_clean)

        # # perform the MAS tail pixel preprocessing method
        # # remove all attributions in the end of the MAS curve where the model response derivative is 0
        # attr_to_clean_ins = remove_pixels(input_tensor.cpu(), attr_to_clean.cpu(), self.model, self.device, total_steps = self.img_hw, img_hw = self.img_hw, segments = segments, type = 1, special_version = special_version)
        # attr_to_clean_del = remove_pixels(input_tensor.cpu(), attr_to_clean.cpu(), self.model, self.device, total_steps = self.img_hw, img_hw = self.img_hw, segments = segments, type = 0, special_version = special_version)
        # attr_to_clean = attr_to_clean_ins + attr_to_clean_del

        # heatmap_overlap(input_tensor, attr_to_clean_del.cpu().numpy(), "clipped")
        # plt.show()
        # plt.close()

        # choose the learning rate heursitically based on the total attribution 
        if lr is None:
            total = torch.sum(torch.abs(torch.sum(attr_to_clean, axis = 2)))
            if total < 10:
                lr = 0.00001
            if total < 500:
                lr = 0.0001
            elif total < 1000: 
                lr = 0.001
            elif total < 10000:
                lr = 0.01
            elif total >= 10000:
                lr = 0.1

        # initialize a neural network with the attribution as its trainable parameters
        net = Net(attr_to_clean)
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # gather the initial loss of the network as a baseline
        insertion, ins_AP = MAS(input_tensor.cpu(), attr_to_clean.cpu(), self.model, self.device, total_steps = self.img_hw, img_hw = self.img_hw, segments = segments, segment_order = segment_order, type = 1, double_loss = 1, special_version = special_version)
        deletion, del_AP = MAS(input_tensor.cpu(), attr_to_clean.cpu(), self.model, self.device, total_steps = self.img_hw, img_hw = self.img_hw, segments = segments, segment_order = segment_order, type = 0, double_loss = 1, special_version = special_version)    
                
        best_loss = (1 - insertion) + deletion 
        # best_loss =  deletion 
        best_attr = attr_to_clean

        for epoch in range(epochs): 
            # Generate a new attribution
            output = net()

            insertion, ins_AP = MAS(input_tensor.cpu(), output.cpu(), self.model, self.device, total_steps = self.img_hw, img_hw = self.img_hw, segments = segments, segment_order = segment_order, type = 1, double_loss = 1, special_version = special_version)
            deletion, del_AP = MAS(input_tensor.cpu(), output.cpu(), self.model, self.device, total_steps = self.img_hw, img_hw = self.img_hw, segments = segments, segment_order = segment_order, type = 0, double_loss = 1, special_version = special_version)    
            loss = (1 - insertion) + deletion
            # loss = deletion

            if loss < best_loss:
                print(loss)
                best_attr = output.detach().clone()
                best_loss = loss

            # optimize the attribution based on the MAS score loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if segments is None:
            return best_attr.cpu().numpy()
        else:
            return best_attr.cpu().numpy(), self.upsize_b(self.downsize(best_attr.permute((2, 0, 1)))).permute((1, 2, 0)).cpu().numpy()