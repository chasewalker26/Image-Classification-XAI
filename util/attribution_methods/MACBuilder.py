import torch
from torchvision import transforms
import numpy as np
from scipy.ndimage import gaussian_filter

from skimage.segmentation import felzenszwalb, slic
from skimage.util import img_as_float

from ..test_methods import MASTestFunctions as MAS

# this code adapts functions from 
# https://github.com/eclique/RISE/blob/master/evaluation.py

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

class RetrieveMAF():
    def __init__(self, model, HW, mode, segments, substrate_fn, device):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            HW (int): image size in pixels given as h*w e.g. 224*224.
            mode (str): 'del' or 'ins'.
            segments (np.ndarray): an N*224*224 array which indicates each of the N segment locations in the image.
            substrate_fn (func): a mapping from old pixels to new pixels.
            device (str): 'cpu' or gpu id e.g. 'cuda:0'.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.HW = HW
        self.mode = mode
        self.segments = segments
        self.substrate_fn = substrate_fn
        self.device = device

    def single_run(self, img_tensor, saliency_map, max_batch_size = 50):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            saliency_map (np.ndarray): saliency map.
            max_batch_size (int): controls the parallelization of the test.
        Return:
            MAF (nd.array): Array containing the magnitude alignment factor of every feature.
            segment_order (np.ndarray): The order of segment indicies ranked by attribution sum high to low.
        """

        n_steps = len(np.unique(self.segments))

        batch_size = n_steps if n_steps < max_batch_size else max_batch_size

        if batch_size > n_steps:
            print("Batch size cannot be greater than number of steps: " + str(n_steps))
            return 0, 0, 0, 0, 0

        # Retrieve softmax score of the original image
        original_pred = self.model(img_tensor.to(self.device)).detach()
        _, index = torch.max(original_pred, 1)
        target_class = index[0]
        percentage = torch.nn.functional.softmax(original_pred, dim = 1)[0]
        original_pred = percentage[target_class].item()

        model_response = np.zeros(n_steps + 1)

        # set the start and stop images for each test
        # get softmax score of the subtrate-applied images
        if self.mode == 'del':
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)

            black_pred = self.model(finish.to(self.device)).detach()
            percentage = torch.nn.functional.softmax(black_pred, dim = 1)[0]
            black_pred = percentage[target_class].item()

            model_response[0] = original_pred
        elif self.mode == 'ins':
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

            blur_pred = self.model(start.to(self.device)).detach()
            percentage = torch.nn.functional.softmax(blur_pred, dim = 1)[0]
            blur_pred = percentage[target_class].item()

            model_response[0] = blur_pred
            
        # find the order of the segments by avg attribution
        segment_saliency = np.zeros(n_steps)
        for i in range(n_steps):
            segment = np.where(self.segments.flatten() == i)[0]
            segment_saliency[i] = np.mean(saliency_map.reshape(self.HW)[segment])

        # segment numbers in order of decreasing saliency
        segment_order = np.flip(np.argsort(segment_saliency), axis = -1)

        density_response = np.zeros(n_steps + 1)
        density_response_deriv = np.zeros(n_steps + 1)

        if self.mode == "del":
            density_response[0] = 1
        elif self.mode == "ins":
            density_response[0] = 0

        total_attr = np.sum(saliency_map.reshape(1, 1, self.HW))

        min_normalized_pred = 1.0
        max_normalized_pred = 0.0

        total_steps = 1
        num_batches = int((n_steps) / batch_size)
        leftover = (n_steps) % batch_size

        batches = np.full(num_batches + 1, batch_size)
        batches[-1] = leftover

        for batch in batches:
            images = torch.zeros((batch, start.shape[1], start.shape[2], start.shape[3]))

            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                segment_coords = np.where(self.segments.flatten() == segment_order[total_steps - 1])[0]

                start.cpu().numpy().reshape(3, self.HW)[:, segment_coords] = finish.cpu().numpy().reshape(3, self.HW)[:, segment_coords]

                images[i] = start

                attr_count = np.sum(saliency_map.reshape(self.HW)[segment_coords]) 
                density_response_deriv[total_steps - 1] = attr_count / total_attr

                if self.mode == "del":
                    density_response[total_steps] = density_response[total_steps - 1] - (attr_count / total_attr)
                elif self.mode == "ins":
                    density_response[total_steps] = density_response[total_steps - 1] + (attr_count / total_attr)

                total_steps += 1

            # get predictions from image batch
            output = self.model(images.to(self.device)).detach()
            percentage = torch.nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class].cpu().numpy()

        # perform monotonic normolization of raw model response
        normalized_model_response = model_response.copy()
        for i in range(n_steps + 1):           
            if self.mode == 'del':
                normalized_pred = (normalized_model_response[i] - black_pred) / (original_pred - black_pred)
                normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
                min_normalized_pred = min(min_normalized_pred, normalized_pred)
                normalized_model_response[i] = min_normalized_pred
            elif self.mode == 'ins':
                normalized_pred = (normalized_model_response[i] - blur_pred) / (original_pred - blur_pred)
                normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
                max_normalized_pred = max(max_normalized_pred, normalized_pred)
                normalized_model_response[i] = max_normalized_pred

        # apply the alignment penalty to the model response
        alignment_penalty = np.abs(normalized_model_response - density_response)
        if self.mode == "ins":
            corrected_scores = normalized_model_response - alignment_penalty
        elif self.mode == "del":
            corrected_scores = normalized_model_response + alignment_penalty    
        
        # scores should be clipped before normalization or else values outside of these bounds will artificially improve the final score
        corrected_scores = corrected_scores.clip(0, 1)
        corrected_scores = (corrected_scores - np.min(corrected_scores)) / (np.max(corrected_scores) - np.min(corrected_scores))        
       

        # score_diff = np.insert(normalized_model_response, 0, 0)
        # score_diff = np.diff(score_diff, 1)

        # # if self.mode == "ins":
        # #     MAF = alignment_penalty
        # # elif self.mode == "del":
        # #     MAF = alignment_penalty

        # AP_diff = np.insert(alignment_penalty, 0, 0)
        # AP_diff = np.diff(AP_diff, 1)
        # # AP_diff = (AP_diff - AP_diff.min()) / (AP_diff.max() - AP_diff.min())
        # # AP_diff[np.where(score_diff == 0)[0]] = 0
        # # AP_diff[np.where(AP_diff < 0)[0]] = 0
        # if self.mode == "ins":
        #     MAF = AP_diff
        # elif self.mode == "del":
        #     MAF = AP_diff

        # # error = (density_response - normalized_model_response)
        # # error = (error - error.min()) / (error.max() - error.min())
        # # error[np.where(score_diff == 0)[0]] = 0
        # # error[np.where(density_response > model_response)[0]] = 0
        # # MAF = error

        # # ER = (normalized_model_response - density_response)
        # # ER[np.where(score_diff == 0)[0]] = 0
        # # above = (2 * ((ER - ER.min()) / (ER.max() - ER.min()))) - 1
        # # below = ((ER - ER.min()) / (ER.max() - ER.min())) - 1
        # # above_loc = np.where(ER > 0)[0]
        # # below_loc = np.where(ER < 0)[0]
        # # ER[above_loc] = above[above_loc]
        # # ER[below_loc] = below[below_loc]
        # # MAF = ER



        # calculate the MAF of each segment by the model and density response derivatives
        dx = 1 # step size
        model_response_deriv = np.gradient(normalized_model_response, dx)
        model_response_deriv_abs = np.abs(model_response_deriv)
        deriv_error = np.abs(model_response_deriv_abs - density_response_deriv)
        diff_too_small = np.where(deriv_error <= 0)[0]
        MAF = np.divide(model_response_deriv_abs, density_response_deriv, out = model_response_deriv_abs.copy(), where = density_response_deriv != 0)
        MAF[diff_too_small] = 0

        return MAF, segment_order, corrected_scores


class Denoise():    
    def __init__(self, model, img_hw, device, substrate_fn, mode = 'ins', segments = None, scale = 100, cutoff = 5):
        r"""Create denoiser instance.
        Args:
            model (nn.Module): Black-box model being explained.
            img_hw: size in pixels of one side of square image e.g. 224
            device: gpu or cpu
            substrate_fn (func): a mapping from old pixels to new pixels.
            mode (str): 'del' or 'ins'.
            scale (int): felzenszwalb segmentation scale.
            cutoff (int): number of times the MAC algorithm iterates while an attribution is not improving.
        """
        self.model = model
        self.img_hw = img_hw
        self.device = device
        self.mode = mode
        self.substrate_fn = substrate_fn
        self.scale = scale
        self.cutoff = cutoff
        self.segments = segments

    # apply the magnitude alignment factor to the segments
    def reduce_noisy_features(self, saliency_map, segments, MAF, segment_order):
        n_steps = len(np.unique(segments))

        # increase the attribution values in each image segment by themself scaled by the segment's MAF
        map = np.ones_like(saliency_map)
        for i in range(n_steps):
            segment_coords = np.where(segments.flatten() == segment_order[i])[0]
            map.reshape(self.img_hw ** 2)[segment_coords] += map.reshape(self.img_hw ** 2)[segment_coords] * MAF[i]
        map = np.reshape(map, (self.img_hw, self.img_hw, 1))
        
        return map

    # perform MAC
    def clean_attribution(self, img_tensor, saliency_map, iterations, max_batch_size = 50):
        new_map = saliency_map
        maps = np.zeros((iterations + 1, new_map.shape[0], new_map.shape[1], new_map.shape[2]))
        scores = np.zeros(iterations + 1)

        maps[0] = new_map
        best_del_score = 1
        best_ins_score = 0
        best_score_index = 0
        stagnant_score_counter = 0
        worse_score_counter = 0

        # generate superpixel masks
        im = np.transpose(img_tensor.squeeze().detach().numpy(), (1, 2, 0))
        img = img_as_float(im)

        if self.segments is not None:
            segments = self.segments
        else:
            # segments = felzenszwalb(img, scale = self.scale, sigma = 0.5, min_size = self.img_hw)
            # segments = felzenszwalb(img, scale = self.scale, sigma = 1, min_size = self.img_hw)
            segments = felzenszwalb(img, scale = 0, sigma = 0.01, min_size = self.img_hw)
            # num_segments = len(np.unique(segments))

            # num_segments = 25
            # segments = slic(img, n_segments=num_segments, compactness=1000, start_label=0)
            # segments = slic(img, n_segments=100, compactness=10, start_label=0)
            # # print("Num Segments = " + str(len(np.unique(segments))))
            # resize_a = transforms.Resize(int(num_segments**(1/2)))
            # resize_b = transforms.Resize(self.img_hw)

        get_MAF = RetrieveMAF(self.model, self.img_hw ** 2, self.mode, segments, substrate_fn = self.substrate_fn, device = self.device)

        # get_score = MAS.MASMetric(self.model, self.img_hw ** 2, self.mode, int(self.img_hw ** 2 / 50), substrate_fn = self.substrate_fn)
        get_score = MAS.MASMetric(self.model, self.img_hw ** 2, self.mode, self.img_hw, substrate_fn = self.substrate_fn)

        for i in range(iterations + 1):
            saliency_map_test = np.abs(np.sum(new_map, axis = 2))

            # # score the current attribution
            _, corrected_score, _, _, _ = get_score.single_run(img_tensor, saliency_map_test, self.device, max_batch_size = max_batch_size)

            # # get the MAF values
            # MAF, segment_order, corrected_score = get_MAF.single_run(img_tensor, saliency_map_test, max_batch_size = max_batch_size)

            score = auc(corrected_score)

            # print(score)

            # track when the score gets better or worse and the index of the best attribution
            if self.mode == 'del':
                if score < best_del_score:
                    best_del_score = score
                    best_score_index = i
                    worse_score_counter = 0
                elif score > best_del_score:
                    worse_score_counter += 1
            elif self.mode == 'ins':
                if score > best_ins_score:
                    best_ins_score = score
                    best_score_index = i
                    worse_score_counter = 0
                elif score < best_ins_score:
                    worse_score_counter += 1

            # track when the score hasn't changed after successive iterations, reset if it does
            if i > 1:
                if round(score, 3) == round(scores[i - 1], 3):
                    stagnant_score_counter += 1
                else:
                    stagnant_score_counter = 0
                    
            scores[i] = score
            # if the score has not been better than the best for <cutoff> iterations in a row, break early and clip arrays
            if stagnant_score_counter == self.cutoff or worse_score_counter == self.cutoff:
                scores = scores[0 : i + 1]
                maps = maps[0 : i + 1]
                iterations = i
                break

            if i == iterations:
                break
            
            # # get the MAF values
            MAF, segment_order, _ = get_MAF.single_run(img_tensor, saliency_map_test, max_batch_size = max_batch_size)
            # modify attribution by MAF values
            map_modifier = self.reduce_noisy_features(saliency_map_test, segments, MAF, segment_order)
            new_map = new_map * map_modifier

            # new_map = resize_b(resize_a(torch.tensor(new_map).permute((2, 0, 1)))).numpy().transpose((1, 2, 0))

            maps[i + 1] = new_map

        return maps[best_score_index], iterations, "start: " + str(round(scores[0], 3)) + " best: " + str(round(scores[best_score_index], 3))