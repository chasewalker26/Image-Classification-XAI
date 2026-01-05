import torch
import torch.nn.functional as F
import numpy as np
from . import lime_image

# https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb

def make_tensor(img):
    return torch.tensor(np.transpose(img, (2, 0, 1)))

def batch_predict(images, model, device):
    model.eval()
    batch = torch.stack(tuple(make_tensor(i) for i in images), dim = 0)

    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def get_lime_attr(img, model, device):
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(img, 
                                            batch_predict, # classification function
                                            model,
                                            device,
                                            top_labels=5, 
                                            hide_color=0, 
                                            num_samples=1000) # number of images that will be sent to classification function

    _, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=False)

    return torch.tensor(mask.reshape(1, mask.shape[0], mask.shape[1])) * torch.ones((3, mask.shape[0], mask.shape[1]))
