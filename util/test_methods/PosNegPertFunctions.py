import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

# this code borrows certain functions from 
# https://github.com/eclique/RISE/blob/master/evaluation.py

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class PositiveNegativePerturbation():
    def __init__(self, model, HW, mode, step_size, substrate_fn):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            HW (int): image size in pixels given as h*w e.g. 224*224.
            mode (str): 'del' (morf), 'ins' or 'lerf'.
            step_size (int): number of pixels modified per one iteration e.g. 224.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['lerf', 'morf']
        self.model = model
        self.HW = HW
        self.mode = mode
        self.step_size = step_size
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, saliency_map, device, patch_mask = None, max_batch_size = 50, CLIP_test_info = None):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            saliency_map (np.ndarray): saliency map.
            device (str): 'cpu' or gpu id e.g. 'cuda:0'.
            max_batch_size (int): controls the parallelization of the test.
        Return:
            n_steps (int): the number of steps used over the test.
            normalized_model_response (nd.array): Array containing the model response at every step.
        """

        if patch_mask is None:
            n_steps = (self.HW + self.step_size - 1) // self.step_size
        else:
            n_steps = len(np.unique(patch_mask))
            self.step_size = int(self.HW / n_steps)

        batch_size = n_steps if n_steps < max_batch_size else max_batch_size

        if batch_size > n_steps:
            print("Batch size cannot be greater than number of steps: " + str(n_steps))
            return 0, 0, 0, 0, 0


        # Set first element of model reponse = 1 for correct prediciton
        model_response = np.zeros(n_steps + 1)

        if CLIP_test_info is None:
            # Retrieve softmax score of the original image
            output = self.model(img_tensor.to(device))
            # if using a huggingface model, we need to access the output logits
            if not isinstance(output, torch.Tensor):
                original_pred = output.logits.detach()
            else:
                original_pred = output.detach()
            _, index = torch.max(original_pred, 1)
            target_class = index[0]
            percentage = torch.nn.functional.softmax(original_pred, dim = 1)[0]
            original_pred = percentage[target_class].item()
            model_response[0] = original_pred

            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
            output = self.model(finish.to(device))
            # if using a huggingface model, we need to access the output logits
            if not isinstance(output, torch.Tensor):
                baseline_pred = output.logits.detach()
            else:
                baseline_pred = output.detach()
            percentage = torch.nn.functional.softmax(baseline_pred, dim = 1)[0]
            baseline_pred = percentage[target_class].item()
        else:
            pred_func = CLIP_test_info["prediction_function"]
            input_tensor = CLIP_test_info["input"]
            CLIP_embeddings = CLIP_test_info["embeddings"]
            target_class, original_pred = pred_func(input_tensor.to(device), self.model, CLIP_embeddings)
            model_response[0] = original_pred

            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
            _, baseline_pred = pred_func(finish.to(device), self.model, CLIP_embeddings)

        # model_response[0] = 1
        # start = img_tensor.clone()
        # finish = self.substrate_fn(img_tensor)

        # model_response[0] = 0
        # start = self.substrate_fn(img_tensor)
        # finish = img_tensor.clone()

        if patch_mask is None:
            # pixels in order of decreasing saliency
            salient_order = np.flip(np.argsort(saliency_map.reshape(-1, self.HW), axis = 1), axis = -1)
        else:
            # patches in order of decreasing saliency
            segment_saliency = np.zeros(n_steps)
            for i in range(n_steps):
                segment = np.where(patch_mask.flatten() == i)[0]
                segment_saliency[i] = np.mean(saliency_map.reshape(self.HW)[segment])

            salient_order = np.flip(np.argsort(segment_saliency, axis = 0), axis = -1)

        # Coordinates of pixels in order of decreasing saliency
        if self.mode == "morf":    
            salient_order = salient_order
        # Coordinates of pixels in order of increasing saliency
        elif self.mode == "lerf":
            salient_order = np.flip(salient_order, axis = -1)

        total_steps = 1
        num_batches = int((n_steps) / batch_size)
        leftover = (n_steps) % batch_size

        if leftover != 0:
            batches = np.full(num_batches + 1, batch_size)
            batches[-1] = leftover
        else:
            batches = np.full(num_batches, batch_size)

        for batch in batches:
            images = torch.zeros((batch, start.shape[1], start.shape[2], start.shape[3]))

            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                if patch_mask is None:
                    coords = salient_order[:, self.step_size * (total_steps - 1) : self.step_size * (total_steps)]
                else:
                    coords = np.where(patch_mask.flatten() == salient_order[total_steps - 1])[0].reshape(1, -1)

                start.cpu().numpy().reshape(1, 3, self.HW)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, self.HW)[0, :, coords]
                images[i] = start
                total_steps += 1

            # get predictions from image batch for VIT or CNN
            if CLIP_test_info is None:
                output = self.model(images.to(device))
                if not isinstance(output, torch.Tensor):
                    output = output.logits.detach()
                else:
                    output = output.detach()
                percentage = torch.nn.functional.softmax(output, dim = 1)
                model_response[total_steps - batch : total_steps] = percentage[:, target_class].cpu().numpy()
            else:
                img_embedding = self.model.encode_image(images.to(device))
                similarities = img_embedding @ CLIP_embeddings.squeeze().T
                model_response[total_steps - batch : total_steps] = torch.nn.functional.softmax(similarities / 0.1, dim=-1)[:, target_class].detach().cpu().numpy()


        min_normalized_pred = 1.0
        max_normalized_pred = 0.0
        # perform monotonic normalization of raw model response
        normalized_model_response = model_response.copy()
        for i in range(n_steps + 1):           
            normalized_pred = (normalized_model_response[i] - baseline_pred) / (abs(original_pred - baseline_pred))
            normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
            min_normalized_pred = min(min_normalized_pred, normalized_pred)
            normalized_model_response[i] = min_normalized_pred

            # normalized_pred = (normalized_model_response[i] - baseline_pred) / (abs(original_pred - baseline_pred))
            # normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
            # max_normalized_pred = max(max_normalized_pred, normalized_pred)
            # normalized_model_response[i] = max_normalized_pred

        return n_steps + 1, model_response
        # return n_steps + 1, normalized_model_response
        # return n_steps + 1, model_response, normalized_model_response
