import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

# this code borrows certain functions from 
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

def pgd_attack(model, images, labels, device, eps=0.3, alpha=2/255, iters=10) :
    images = images.to(device)
    labels = labels.to(device)
    loss = torch.nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min = -eps, max = eps)
        images = torch.clamp(ori_images + eta, min = 0, max = 1).detach_()
            
    return images

class MASMetric():
    def __init__(self, model, HW, mode, step_size, substrate_fn):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            HW (int): image size in pixels given as h*w e.g. 224*224.
            mode (str): 'del' (morf), 'ins' or 'lerf'.
            step_size (int): number of pixels modified per one iteration e.g. 224.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins', 'lerf', 'morf']
        self.model = model
        self.HW = HW
        self.mode = mode
        self.step_size = step_size
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, saliency_map, device, patch_mask = None, max_batch_size = 50, special_version = False, return_embeddings = False, CLIP_test_info = None):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            saliency_map (np.ndarray): saliency map.
            device (str): 'cpu' or gpu id e.g. 'cuda:0'.
            patch_mask (Tensor): A mask which indicates N patch locations in range [0, N-1].
            max_batch_size (int): controls the parallelization of the test.
        Return:
            n_steps (int): the number of steps used over the test.
            corrected_scores (nd.array): Array containing MAS scores at every step.
            alignment_penalty (nd.array): Array containing alignment penalty at every step.
            density_response (nd.array): Array containing the density response at every step.
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

        model_response = np.zeros(n_steps + 1)
        entropy = np.ones(n_steps + 1)

        embeddings = []
        classes = []

        if CLIP_test_info is None:
            # Retrieve softmax score of the original image
            output = self.model(img_tensor.to(device))
            # if using a huggingface model, we need to access the output logits
            if not isinstance(output, torch.Tensor):
                original_pred = output.logits.detach()
            else:
                original_pred = output.detach()
            _, index = torch.max(original_pred, 1)
            orig_class = index
            target_class = index[0]
            percentage = torch.nn.functional.softmax(original_pred, dim = 1)[0]
            original_pred = percentage[target_class].item()


            if return_embeddings == True:
                num_blocks = len(self.model.blocks)
                _, n_patches, embed_dim = self.model.blocks[-1].get_block_out().detach().shape

                orig_embedding = torch.empty((num_blocks, 1, n_patches, embed_dim))
                counter = 0
                for block in self.model.blocks:
                    orig_embedding[counter] = block.get_block_out().detach()
                    counter += 1
                
                if len(orig_embedding.shape) != 4:
                    orig_embedding = torch.unsqueeze(orig_embedding, 1)


            # set the start and stop images for each test
            # get softmax score of the subtrate-applied images
            if self.mode == 'del' or self.mode == 'morf' or self.mode == 'lerf':
                start = img_tensor.clone()
                finish = self.substrate_fn(img_tensor)

                output = self.model(finish.to(device))
                # if using a huggingface model, we need to access the output logits
                if not isinstance(output, torch.Tensor):
                    baseline_pred = output.logits.detach()
                else:
                    baseline_pred = output.detach()
                baseline_percentage = torch.nn.functional.softmax(baseline_pred, dim = 1)[0]
                baseline_pred = baseline_percentage[target_class].item()

                output = self.model(start.to(device))
                # if using a huggingface model, we need to access the output logits
                if not isinstance(output, torch.Tensor):
                    normal_pred = output.logits.detach()
                else:
                    normal_pred = output.detach()
                normal_percentage = torch.nn.functional.softmax(normal_pred, dim = 1)[0]
                normal_pred = normal_percentage[target_class].item()

                model_response[0] = original_pred
                entropy[0] = -torch.sum(percentage * torch.log2(percentage), dim=-1).cpu().numpy()

            elif self.mode == 'ins':
                start = self.substrate_fn(img_tensor)
                finish = img_tensor.clone()

                output = self.model(start.to(device))
                # if using a huggingface model, we need to access the output logits
                if not isinstance(output, torch.Tensor):
                    baseline_pred = output.logits.detach()
                else:
                    baseline_pred = output.detach()                
                baseline_percentage = torch.nn.functional.softmax(baseline_pred, dim = 1)[0]
                baseline_pred = baseline_percentage[target_class].item()

                output = self.model(finish.to(device))
                # if using a huggingface model, we need to access the output logits
                if not isinstance(output, torch.Tensor):
                    normal_pred = output.logits.detach()
                else:
                    normal_pred = output.detach()                
                normal_percentage = torch.nn.functional.softmax(normal_pred, dim = 1)[0]
                normal_pred = normal_percentage[target_class].item()

                model_response[0] = baseline_pred
                entropy[0] = -torch.sum(baseline_percentage * torch.log2(baseline_percentage), dim=-1).cpu().numpy()

        else:
            # Retrieve softmax score of the original image
            pred_func = CLIP_test_info["prediction_function"]
            input_tensor = CLIP_test_info["input"]
            CLIP_embeddings = CLIP_test_info["embeddings"]
            target_class, original_pred = pred_func(input_tensor.to(device), self.model, CLIP_embeddings)

            # set the start and stop images for each test
            # get softmax score of the subtrate-applied images
            if self.mode == 'del':
                start = img_tensor.clone()
                finish = self.substrate_fn(img_tensor)
                _, baseline_pred = pred_func(finish.to(device), self.model, CLIP_embeddings)
                model_response[0] = original_pred
            elif self.mode == 'ins':
                start = self.substrate_fn(img_tensor)
                finish = img_tensor.clone()
                _, baseline_pred = pred_func(start.to(device), self.model, CLIP_embeddings)
                model_response[0] = baseline_pred

        if patch_mask is None:
            # pixels in order of decreasing saliency
            salient_order = np.flip(np.argsort(saliency_map.reshape(-1, self.HW), axis = 1), axis = -1)
            
            if self.mode == 'lerf':
                salient_order = np.argsort(saliency_map.reshape(-1, self.HW), axis = 1)
        else:
            # patches in order of decreasing saliency
            segment_saliency = np.zeros(n_steps)
            for i in range(n_steps):
                segment = np.where(patch_mask.flatten() == i)[0]
                segment_saliency[i] = np.mean(saliency_map.reshape(self.HW)[segment])

            salient_order = np.flip(np.argsort(segment_saliency, axis = 0), axis = -1)

            if self.mode == 'lerf':
                salient_order = np.argsort(segment_saliency, axis = 0)

        density_response = np.zeros(n_steps + 1)

        if self.mode == "del" or self.mode == 'morf' or self.mode == 'lerf':
            density_response[0] = 1
        elif self.mode == "ins":
            density_response[0] = 0

        total_attr = np.sum(saliency_map.reshape(1, 1, self.HW))

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

                attr_count = np.sum(saliency_map.reshape(1, 1, self.HW)[0, :, coords]) 
                if self.mode == "del" or self.mode == "morf" or self.mode == "lerf":
                    density_response[total_steps] = density_response[total_steps - 1] - (attr_count / total_attr)
                elif self.mode == "ins":
                    density_response[total_steps] = density_response[total_steps - 1] + (attr_count / total_attr)

                total_steps += 1

            # get predictions from image batch for VIT or CNN
            if CLIP_test_info is None:
                output = self.model(images.to(device))
                if not isinstance(output, torch.Tensor):
                    output = output.logits.detach()
                else:
                    output = output.detach()
                percentage = torch.nn.functional.softmax(output, dim = 1)
                entropy[total_steps - batch : total_steps] = -torch.sum(percentage * torch.log2(percentage), dim=-1).cpu().numpy()
                model_response[total_steps - batch : total_steps] = percentage[:, target_class].cpu().numpy()
            # get predictions from image batch for CLIP
            else:
                img_embedding = self.model.encode_image(images.to(device))
                similarities = img_embedding @ CLIP_embeddings.squeeze().T
                model_response[total_steps - batch : total_steps] = torch.nn.functional.softmax(similarities / 0.1, dim=-1)[:, target_class].detach().cpu().numpy()

            if return_embeddings == True:
                classes.append(torch.max(output, 1)[1])

                attn_mask_embeddings = torch.empty((num_blocks, batch, n_patches, embed_dim))
                counter = 0
                for block in self.model.blocks:
                    attn_mask_embeddings[counter] = block.get_block_out().detach()
                    counter += 1
                                    
                if len(attn_mask_embeddings.shape) != 4:
                    attn_mask_embeddings = torch.unsqueeze(attn_mask_embeddings, 1)
                                                                                
                embeddings.append(attn_mask_embeddings)

        min_normalized_pred = 1.0
        max_normalized_pred = 0.0
        # perform monotonic normalization of raw model response
        normalized_model_response = model_response.copy()
        for i in range(n_steps + 1):           
            normalized_pred = (normalized_model_response[i] - baseline_pred) / (abs(original_pred - baseline_pred))
            normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
            if self.mode == 'del' or self.mode == 'morf' or self.mode == 'lerf':
                min_normalized_pred = min(min_normalized_pred, normalized_pred)
                normalized_model_response[i] = min_normalized_pred
            elif self.mode == 'ins':
                max_normalized_pred = max(max_normalized_pred, normalized_pred)
                normalized_model_response[i] = max_normalized_pred

        if special_version == True:
            n = len(normalized_model_response)

            # Objective function components
            Q = matrix(2 * np.eye(n))  # 2 * I because the objective function is (y - x)^2
            c = matrix(-2 * normalized_model_response)  # -2 * y for the objective function -2 * (y - x)

            # Derivative Constraints: 
            # x_2 - x_1 <= x_3 - x_2 (deletion)
            # x_2 - x_1 >= x_3 - x_2 (insertion)
            A_ineq = np.zeros((n - 2, n))
            b_ineq = np.full(n - 2, 0)
            row_indices = np.arange(n - 2)
            if self.mode == 'del':
                A_ineq[row_indices, row_indices] = -1
                A_ineq[row_indices, row_indices + 1] = 2
                A_ineq[row_indices, row_indices + 2] = -1
            elif self.mode == 'ins':
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
            b_eq = matrix(np.array([normalized_model_response[0], normalized_model_response[-1]]), (2, 1), 'd')

            # Solve and extract solution
            sol = solvers.qp(Q, c, G, h, A_eq, b_eq)
            normalized_model_response = np.squeeze(np.array(sol['x']))

        alignment_penalty = np.abs(normalized_model_response - density_response)

        if self.mode == "del" or self.mode == 'morf' or self.mode == 'lerf':
            corrected_scores = normalized_model_response + alignment_penalty    
        elif self.mode == "ins":
            corrected_scores = normalized_model_response - alignment_penalty

        # scores should be clipped before normalization or else values outside of these bounds will artificially improve the final score
        corrected_scores = corrected_scores.clip(0, 1)
        corrected_scores = (corrected_scores - np.min(corrected_scores)) / (np.max(corrected_scores) - np.min(corrected_scores))

        # if somehow the blurred or black image recieved the same prediction as the input image, causing a failure of the above line, assign the attribution an ROC with a score of 0.5
        if np.isnan(corrected_scores).any():
            if self.mode == 'del' or self.mode == 'morf':
                corrected_scores = np.linspace(1, 0, n_steps + 1)
            elif self.mode == 'ins' or self.mode == 'lerf':
                corrected_scores = np.linspace(0, 1, n_steps + 1)
                
        if return_embeddings == True:
            if self.mode == "morf" or self.mode == 'del' or self.mode == "lerf":
                embeddings.insert(0, orig_embedding)
                classes.insert(0, orig_class)
            elif self.mode == 'ins':
                classes.append(orig_class)
                embeddings.append(orig_embedding)
                
            embeddings = torch.cat(embeddings, axis = 1)
            classes = torch.cat(classes, axis = 0)

            return embeddings.cpu().numpy(), classes.cpu().numpy(), model_response, salient_order


        # return n_steps + 1, corrected_scores, entropy, density_response, model_response
        return n_steps + 1, corrected_scores, entropy, density_response, normalized_model_response
