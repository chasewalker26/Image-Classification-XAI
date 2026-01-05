import numpy as np
import torch
import torchvision.transforms as transforms
from numpy import *
import argparse
import os
import csv
import warnings
from utils.metrices import *
import time
from PIL import Image
from collections import Counter
from transformers import CLIPTokenizerFast

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

# utils
from util import model_utils

# models
from torchvision import models
from util.modified_models import resnet

import clip
from util.attribution_methods.CLIP.Game_MM_CLIP import clip as mm_clip
from util.attribution_methods.CLIP.CLIP_Surgery import clip as surgery_clip
from util.attribution_methods.CLIP.CLIP_lrp.CLIP.clip import clip as lrp_clip
from util.attribution_methods.CLIP.M2IB.scripts.clip_wrapper import ClipWrapper

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch16_224
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_base_patch16_224 as vit_base_patch16_224_LRP

    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch32_224
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_base_patch32_224 as vit_base_patch32_224_LRP

# attribution methods
from util.attribution_methods import saliencyMethods as attr
from util.attribution_methods.lime import limeAttr
from util.attribution_methods import GIGBuilder as GIG_Builder
from util.attribution_methods import AGI as AGI
from captum.attr import GuidedBackprop, LayerGradCam, GradientShap, GuidedGradCam, FeatureAblation, Occlusion
from util.attribution_methods import XRAIBuilder as XRAI

from util.attribution_methods.VIT_LRP.ViT_explanation_generator import Baselines, LRP
from util.attribution_methods.ViT_CX.ViT_CX import ViT_CX
from util.attribution_methods.TIS import TIS
from util.attribution_methods import MDAFunctions

from util.attribution_methods.CLIP.generate_emap import imgprocess_keepsize, mm_interpret, \
        clip_encode_dense, grad_eclip, mask_clip, compute_rollout_attention, \
        clip_surgery_map, m2ib_clip_map,  clip_lrp

# evaluation metrics
from util.test_methods import MASTestFunctions as MAS
from scipy.stats import spearmanr as spr
from skimage.metrics import structural_similarity as ssim
from skimage import feature

with open('../../util/class_maps/ImageNet/ILSVRC2012_validation_ground_truth.txt') as f:
    gnd_truth = [int(line.strip()) for line in f.readlines()]
with open('../../util/class_maps/ImageNet/map_clsloc.txt') as f:
    class_map = [line.strip() for line in f.readlines()] 
with open('../../util/class_maps/ImageNet/imagenet_classes.txt') as f:
    class_list = [line.strip() for line in f.readlines()]

def normalize_image(x):
    x = np.array(x).astype(np.float32)

    if (x.max() - x.min()) == 0:
        return x

    # fix any outliers
    x[x == float("Inf")] = 0
    x[x == float("-Inf")] = 0

    # normalize
    return (x - x.min()) / (x.max() - x.min())

# evaluate the attr for normal network against attr for randomized network
def get_sanity(normal_attr, random_layer_attr, abs = False):
    normal_attr_0_1 = normalize_image(normal_attr)
    random_layer_attr_0_1 = normalize_image(random_layer_attr)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if abs == False:
            ssim_val = ssim(normal_attr_0_1, random_layer_attr_0_1, gaussian_weights = True, channel_axis = 2)
            normal_hog = feature.hog(normal_attr_0_1, pixels_per_cell = (16, 16), channel_axis = 2)
            rand_hog = feature.hog(random_layer_attr_0_1, pixels_per_cell = (16, 16), channel_axis = 2)
        else:
            ssim_val = ssim(normal_attr_0_1, random_layer_attr_0_1, gaussian_weights = True)
            normal_hog = feature.hog(normal_attr_0_1, pixels_per_cell = (16, 16))
            rand_hog = feature.hog(random_layer_attr_0_1, pixels_per_cell = (16, 16))

    spr_val, _ = spr(normal_attr.flatten(), random_layer_attr.flatten())
    hog_val, _ = spr(normal_hog, rand_hog)

    sanity_result_counter = Counter({
        "SSIM": ssim_val,
        "SPR": spr_val,
        "HOG": hog_val
    })

    return sanity_result_counter

def randomize_CNN_model(model):
    for layer_name, layer in model.named_modules():  # Iterate over layers
        if isinstance(layer, torch.nn.Module) and len(list(layer.parameters())) > 0:

            # Apply randomization to all parameters in the layer
            with torch.no_grad():
                # Initialize weights using Kaiming/Xavier depending on layer type
                if isinstance(layer, torch.nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                elif isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)

    return model

def randomize_VIT_model(model, randomization_fn=torch.nn.init.normal_):
    for layer_name, layer in model.named_modules():  # Iterate over layers
        if isinstance(layer, torch.nn.Module) and len(list(layer.parameters())) > 0:
            # Apply randomization to all parameters in the layer
            with torch.no_grad():
                for param in layer.parameters():
                    randomization_fn(param)

    return model

def randomize_CLIP_model(model, randomization_fn=torch.nn.init.normal_):
    # Randomize each layer one at a time
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                randomization_fn(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)  # Optionally zero the bias

        elif isinstance(module, torch.nn.Embedding):
            with torch.no_grad():
                randomization_fn(module.weight)

    return model


def get_CLIP_pred(input_tensor, model, all_classes_embedding):
    img_embedding = model.encode_image(input_tensor)
    similarities = img_embedding @ all_classes_embedding.squeeze().T
    pred_class = similarities.argmax().item()
    prediction = torch.nn.functional.softmax(similarities / 0.1, dim=-1)[:, pred_class].item()
    return pred_class, prediction

def get_classifier_pred(input_tensor, model, device):
    pred_class = model_utils.getClass(input_tensor, model, device)
    prediction = model_utils.getPrediction(input_tensor, model, device, pred_class)[0]
    return pred_class, prediction

def get_CNN_attr(input_tensor, trans_img, target_class, testing_dict, random = False):
    if random == False:
        model = testing_dict["models"][0]
        modified_model = testing_dict["models"][1]
    else:
        model = testing_dict["models"][2]
        modified_model = testing_dict["models"][3]

    batch_size = testing_dict["batch_size"]
    img_hw = testing_dict["img_hw"]
    device = testing_dict["device"]
    attr_function = testing_dict["attr_func"]
    steps = 50
    baseline = 0

    resize = transforms.Resize((img_hw, img_hw), antialias = True)

    num_patches = 14
    downsize = transforms.Resize((num_patches, num_patches), interpolation=transforms.InterpolationMode.NEAREST_EXACT)
    patch_ids = torch.arange(num_patches ** 2).reshape((num_patches, num_patches))
    patch_mask = patch_ids.repeat_interleave(int(img_hw / num_patches), dim=0).repeat_interleave(int(img_hw / num_patches), dim=1).to(device)

    if attr_function == "grad":
        input_tensor.requires_grad = True
        saliency_map, _ = attr.getGradientsParallel(input_tensor.to(device), model, target_class)
        input_tensor.requires_grad = False
    elif attr_function == "inp_x_grad":
        input_tensor.requires_grad = True
        grad, _ = attr.getGradientsParallel(input_tensor.to(device), model, target_class)
        input_tensor.requires_grad = False
        saliency_map = input_tensor.squeeze() * grad
    elif attr_function == "ig":
        saliency_map = attr.IG(input_tensor, model, steps, batch_size, 1, baseline, device, target_class)
    elif attr_function == "lig":
        saliency_map = attr.IG(input_tensor, model, steps, batch_size, .9, baseline, device, target_class)
    elif attr_function == "idg":
        saliency_map = attr.IDG(input_tensor, model, steps, batch_size, baseline, device, target_class)
    elif attr_function == "gig":
        call_model_args = {'class_idx_str': target_class.item()}
        guided_ig = GIG_Builder.GuidedIG()
        baseline = torch.zeros_like(input_tensor.cpu())
        saliency_map = guided_ig.GetMask(input_tensor.cpu(), model, device, GIG_Builder.call_model_function, call_model_args, x_baseline=baseline, x_steps=steps, max_dist=1.0, fraction=0.5).squeeze()
    elif attr_function == "agi":
        epsilon = 0.05
        topk = 1
        max_iter = 20
        agi_img = trans_img.permute(1, 2, 0).numpy().astype(np.float32)
        mean = testing_dict["normalize"].mean 
        std = testing_dict["normalize"].std
        norm_layer = AGI.Normalize(mean, std)
        agi_model = torch.nn.Sequential(norm_layer, model).to(device)        
        selected_ids = range(0, 999, int(1000 / topk))
        _, _, agi = AGI.test(agi_model, device, agi_img, epsilon, topk, selected_ids, max_iter)
        percentile = 80
        upperbound = 99
        hm = agi
        hm = np.mean(hm, axis=0)
        q = np.percentile(hm, percentile)
        u = np.percentile(hm, upperbound)
        hm[hm<q] = q
        hm[hm>u] = u
        hm = (hm-q)/(u-q)
        saliency_map = torch.from_numpy(np.reshape(hm, (1, img_hw, img_hw)))
    elif attr_function == "sg":
        saliency_map = attr.smoothGrad("IG", input_tensor, model, 50, baseline, target_class, device)
    elif attr_function == "xrai":
        xrai_object = XRAI.XRAI()
        ig = attr.IG(input_tensor, model, steps, batch_size, 1, baseline, device, target_class)
        saliency_map = xrai_object.GetMask(input_tensor.squeeze().permute(1, 2, 0).cpu(), base_attribution = ig.permute(1, 2, 0).cpu().detach()).reshape((img_hw, img_hw, 1))
        saliency_map = torch.from_numpy(saliency_map).permute(2, 0, 1)
    elif attr_function == "gc":
        layer = model.layer4
        layer_gc = LayerGradCam(model, layer)
        input_tensor.requires_grad = True
        gc = layer_gc.attribute(input_tensor.to(device), target_class, relu_attributions=True)
        input_tensor.requires_grad = False
        saliency_map = resize(gc.squeeze().reshape((1, 7, 7))).reshape((1, img_hw, img_hw)).cpu() * torch.ones((3, img_hw, img_hw))
    elif attr_function == "gbp":
        guided_bp = GuidedBackprop(modified_model)
        input_tensor.requires_grad = True
        saliency_map = guided_bp.attribute(input_tensor.to(device), target = target_class).squeeze()
        input_tensor.requires_grad = False
    elif attr_function == "ggc":
        guided_gc = GuidedGradCam(modified_model, modified_model.layer4)
        input_tensor.requires_grad = True
        saliency_map = guided_gc.attribute(input_tensor.to(device), target_class).squeeze()
        input_tensor.requires_grad = False
    elif attr_function == "gs":
        baselines = torch.randn(1, 3, 224, 224)
        gradient_shap = GradientShap(model)
        saliency_map = gradient_shap.attribute(input_tensor.to(device), baselines.to(device), target = target_class).squeeze()
    elif attr_function == "lime":
        img_lime = trans_img.permute(1, 2, 0).numpy().astype(np.float32) 
        saliency_map = limeAttr.get_lime_attr(img_lime, model, device)
    elif attr_function == "fa":
        ablator = FeatureAblation(model)
        saliency_map = resize(downsize(ablator.attribute(input_tensor.to(device), target=target_class, feature_mask=patch_mask))).squeeze()
    elif attr_function == "occ":
        ablator = Occlusion(model)
        saliency_map = resize(downsize(ablator.attribute(input_tensor.to(device), target=target_class, sliding_window_shapes=(3,64,64), strides=32))).squeeze()
    else:
        print("Model-attribution mismatch, please use --help.")
        exit()

    return np.abs(np.sum(saliency_map.detach().cpu().numpy(), axis = 0, keepdims = True)).reshape((224, 224, 1))

def get_VIT_attr(input_tensor, trans_img, target_class, testing_dict, random = False):
    if random == False:
        model = testing_dict["models"][0]
        lrp_model = testing_dict["models"][1]
    else:
        model = testing_dict["models"][2]
        lrp_model = testing_dict["models"][3]

    num_patches = testing_dict["num_patches"]
    img_hw = testing_dict["img_hw"]
    device = testing_dict["device"]
    attr_function = testing_dict["attr_func"]

    resize = transforms.Resize((img_hw, img_hw), antialias = True)
    resize_square = transforms.Resize((img_hw, img_hw), antialias = True, interpolation=transforms.InterpolationMode.NEAREST_EXACT)

    explainer = Baselines(model)
    LRP_explainer = LRP(lrp_model)

    if attr_function == "attn":
        saliency_map = explainer.generate_raw_attn(input_tensor.to(device), device)
        saliency_map = resize(saliency_map.cpu().detach()).permute(1, 2, 0)
    elif attr_function == "grad":
        saliency_map = explainer.generate_grad(input_tensor.to(device), target_class, device)
        saliency_map = resize(saliency_map.cpu().detach()).permute(1, 2, 0)
    elif attr_function == "n_rollout":
        saliency_map, _, _ = explainer.generate_naive_rollout(input_tensor.to(device))
        saliency_map = resize(saliency_map.cpu().detach()).permute(1, 2, 0)
    elif attr_function == "rollout":
        saliency_map, _, _ = explainer.generate_rollout(input_tensor.to(device))
        saliency_map = resize(saliency_map.cpu().detach()).permute(1, 2, 0)
    elif attr_function == "t_attn":
        _, _, saliency_map, _, _ = explainer.generate_transition_attention_maps(input_tensor.to(device), target_class, start_layer = 0, device = device)
        saliency_map = resize(saliency_map.cpu().detach()).permute(1, 2, 0)
    elif attr_function == "bi_attn":
        saliency_map, _ = explainer.bidirectional(input_tensor.to(device), target_class, device = device)
        saliency_map = resize(saliency_map.cpu().detach()).permute(1, 2, 0)
    elif attr_function == "t_attr":
        saliency_map = LRP_explainer.generate_LRP(input_tensor.to(device), target_class, method="transformer_attribution", start_layer = 0, device = device)
        saliency_map = resize(saliency_map.cpu().detach()).permute(1, 2, 0)
    elif attr_function == "VIT_CX":
        target_layer = model.blocks[-1].norm1
        result, _ = ViT_CX(model, input_tensor, target_layer, gpu_batch=1, device = device)
        saliency_map = (result.reshape((img_hw, img_hw, 1)) * torch.ones((img_hw, img_hw, 3)))
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    elif attr_function == "TIS":
        saliency_method = TIS(model, batch_size=64)
        saliency_map = saliency_method(input_tensor.to(device), class_idx=target_class).cpu()
        saliency_map = resize(saliency_map.reshape((-1, saliency_map.shape[0], saliency_map.shape[1])).detach()).permute(1, 2, 0)
    elif attr_function == "InFlow":
        saliency_map, _ = explainer.generate_RAVE(input_tensor.to(device), target_class, option='b', device = device)
        saliency_map = resize(saliency_map.cpu().detach()).permute(1, 2, 0)
    elif attr_function == "MDA":
        klen = 31
        ksig = 31
        kern = MAS.gkern(klen, ksig)
        blur = lambda x: torch.nn.functional.conv2d(x, kern, padding = klen // 2)
        blur_perc = model_utils.getPrediction(blur(input_tensor.cpu()), model, device, target_class)[0] * 100
        while blur_perc > 1:
            klen += 4
            ksig += 4
            kern = MAS.gkern(klen, ksig)
            blur = lambda x: torch.nn.functional.conv2d(x, kern, padding = klen // 2)
            blur_perc = model_utils.getPrediction(blur(input_tensor.cpu()), model, device, target_class)[0] * 100

            if klen > 101:
                break

        bi_attn, _ = explainer.bidirectional(input_tensor.to(device), target_class, device=device)
        bi_attn = resize(bi_attn.cpu().detach()).permute(1, 2, 0).cpu().numpy() * np.ones((img_hw, img_hw, 3))

        mda, _, _ = MDAFunctions.MDA(trans_img, input_tensor.cpu(), bi_attn, num_patches ** 2, blur, model, device, img_hw, max_batch_size = 5)
        saliency_map = torch.from_numpy(mda)
    else:
        print("Model-attribution mismatch, please use --help.")
        exit()

    return np.abs(np.sum(saliency_map.detach().cpu().numpy(), axis = 2, keepdims=True))

def get_CLIP_attr(trans_img, target_class, testing_dict, random = False):
    if random == False:
        clipmodel = testing_dict["models"][0]
        mm_lrp_clipmodel = testing_dict["models"][1]
        surgery_model = testing_dict["models"][2]
        m2ib_model = testing_dict["models"][3]
        clip_tokenizer = testing_dict["models"][4]
    else:
        clip_tokenizer = testing_dict["models"][4]
        clipmodel = testing_dict["models"][5]
        mm_lrp_clipmodel = testing_dict["models"][6]
        surgery_model = testing_dict["models"][7]
        m2ib_model = testing_dict["models"][8]
    
    preprocess = testing_dict["normalize"]
    device = testing_dict["device"]
    img_hw = testing_dict["img_hw"]
    num_patches = testing_dict["num_patches"]
    attr_function = testing_dict["attr_func"]

    resize = transforms.Resize((img_hw, img_hw), antialias = True)
    resize_square = transforms.Resize((img_hw, img_hw), antialias = True, interpolation=transforms.InterpolationMode.NEAREST_EXACT)

    caption = "a photo of a " + class_list[target_class]
    txts = [caption]
    text_processed = clip.tokenize(txts).to(device)
    txt_embedding = clipmodel.encode_text(text_processed)
    txt_embedding = torch.nn.functional.normalize(txt_embedding, dim=-1)

    img = transforms.functional.to_pil_image(trans_img)
    img_keepsized = imgprocess_keepsize(img).to(device).unsqueeze(0)
    outputs, v_final, last_input, v, q_out, k_out,\
        attn, att_output, map_size = clip_encode_dense(img_keepsized, clipmodel)
    img_embedding = torch.nn.functional.normalize(outputs[:,0], dim=-1)
    cosines = (img_embedding @ txt_embedding.T)[0]

    if attr_function == "eclip":
        emap = [grad_eclip(c, q_out, k_out, v, att_output, map_size, withksim=True) for c in cosines]
        emap = torch.stack(emap, dim=0).sum(0)  
    elif attr_function == "eclip_wo":
        emap = [grad_eclip(c, q_out, k_out, v, att_output, map_size, withksim=False) for c in cosines]
        emap = torch.stack(emap, dim=0).sum(0)  
    elif attr_function == "game":
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        text_tokenized = mm_clip.tokenize(txts).to(device)
        emap = mm_interpret(model=mm_lrp_clipmodel, image=img_clipreprocess, texts=text_tokenized, device=device)    
        emap = emap.sum(0) 
    elif attr_function == "maskclip":
        emap = mask_clip(txt_embedding.T, v_final, k_out, map_size)
        emap = emap.sum(0)
    elif attr_function == "rollout":
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        text_tokenized = mm_clip.tokenize(txts).to(device)
        attentions = mm_interpret(model=mm_lrp_clipmodel, image=img_clipreprocess, texts=text_tokenized, device=device, rollout=True)      
        emap = compute_rollout_attention(attentions)[0]
    elif attr_function == "selfattn":
        emap = attn[0,:1,1:].detach().reshape(*map_size)
    elif attr_function == "surgery":
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        all_texts = ['airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']
        all_texts = txts + all_texts
        emap = clip_surgery_map(model=surgery_model, image=img_clipreprocess, texts=all_texts, device=device)[0,:,:,0]
    elif attr_function == "m2ib":
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        emap = m2ib_clip_map(model=m2ib_model, clip_tokenizer=clip_tokenizer, image=img_clipreprocess, texts=txts[0], device=device)
        emap = torch.tensor(emap)
    elif attr_function == "lrp":
        img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
        _, hm = clip_lrp(img_clipreprocess, lrp_clip.tokenize(txts[0]).to(device), mm_lrp_clipmodel, device)
        emap = hm.reshape((-1, num_patches, num_patches))
    else:
        print("Model-attribution mismatch, please use --help.")
        exit()

    if attr_function != "MDA":
        saliency_map = resize(emap.unsqueeze(0))[0].squeeze().reshape((img_hw, img_hw, -1))

    return np.abs(np.sum(saliency_map.detach().cpu().numpy(), axis = 2, keepdims=True))


def evaluate_sanity(testing_dict):
    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + testing_dict["model_name"] + ".txt").astype(np.int64)

    script_start = time.time()

    images_used = 0
    # look at test images in order from 1
    for image in sorted(os.listdir(testing_dict["imagenet_dataset"])):    
        if (images_used) % 10 == 0:
            loop_start_time = time.time()

        if images_used == testing_dict["image_count"]:
            print("method finished")
            break

        # check if the current image is an invalid image for testing, 0 indexed
        image_num = int((image.split("_")[2]).split(".")[0]) - 1
        # check if the current image is an invalid image for testing
        if correctly_classified[image_num] == 0:
            continue

        image_path = testing_dict["imagenet_dataset"] + "/" + image
        PIL_img = Image.open(image_path)

        # put the image in form needed for prediction
        trans_img = testing_dict["transform"](PIL_img)

        # only rgb images can be classified
        if trans_img.shape != (3, testing_dict["img_hw"], testing_dict["img_hw"]):
            continue

        # Get the class and prediciton for the image
        if "CLIP" in testing_dict["model_name"]:
            input_tensor = testing_dict["normalize"](transforms.functional.to_pil_image(trans_img))
            input_tensor = torch.unsqueeze(input_tensor, 0).to(testing_dict["device"])
            target_class, _ = get_CLIP_pred(input_tensor, testing_dict["models"][0], testing_dict["embeddings"][0])

            target_class_rand, _ = get_CLIP_pred(input_tensor, testing_dict["models"][5], testing_dict["embeddings"][1])
        else:
            input_tensor = testing_dict["normalize"](trans_img)
            input_tensor = torch.unsqueeze(input_tensor, 0).to(testing_dict["device"])
            target_class, _ = get_classifier_pred(input_tensor, testing_dict["models"][0], testing_dict["device"])

            target_class_rand, _ = get_classifier_pred(input_tensor, testing_dict["models"][2], testing_dict["device"])

        if (images_used + 1) % 10 == 0:
            print(testing_dict["model_name"] + " " + testing_dict["attr_func"] + " sanity tests, image: " + image + " " + str(images_used + 1) + "/" + str(testing_dict["image_count"]))
        
        if "R" in testing_dict["model_name"]:
            attribution = get_CNN_attr(input_tensor, trans_img, target_class, testing_dict, random = False)
            rand_attribution = get_CNN_attr(input_tensor, trans_img, target_class_rand, testing_dict, random = True)
        elif "VIT" in testing_dict["model_name"]:
            attribution = get_VIT_attr(input_tensor, trans_img, target_class, testing_dict, random = False)
            rand_attribution = get_VIT_attr(input_tensor, trans_img, target_class_rand, testing_dict, random = True)
        elif "CLIP" in testing_dict["model_name"]:
            attribution = get_CLIP_attr(trans_img, target_class, testing_dict, random = False)
            rand_attribution = get_CLIP_attr(trans_img, target_class_rand, testing_dict, random = True)
                
        # evaluate the attribution
        if images_used == 0:
            sanity_result_counter = get_sanity(attribution, rand_attribution)
        else:
            sanity_result_counter += get_sanity(attribution, rand_attribution)    

        images_used += 1

        if (images_used) % 10 == 0:
                print(time.time() - loop_start_time)

    total_time = time.time() - script_start

    # make the test folder if it doesn't exist
    folder = "sanity_test_results/" + testing_dict["model_name"] + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save all data
    file_name = testing_dict["attr_func"] + "_" + str(testing_dict["image_count"]) + "_images"
    with open(folder + file_name + ".csv", 'w') as f:
        write = csv.writer(f)
        for i, k in enumerate(sanity_result_counter):
            write.writerow([k, str(sanity_result_counter[k] / images_used)])
        
        write.writerow(["Total Runtime", str(total_time)])

    return

def main(args):
    img_hw = 224
    device = 'cuda:' + str(args.cuda_num) if torch.cuda.is_available() else 'cpu'

    # set up models 
    if args.model == "R101":
        model = models.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2").to(device)
        modified_model = resnet.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2").to(device)
        model_rand = randomize_CNN_model(models.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2").to(device))
        modified_model_rand = randomize_CNN_model(resnet.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2").to(device))
        img_hw = 224
        num_patches = 0
        batch_size = 50
        model_list = [model, modified_model, model_rand, modified_model_rand]
    elif args.model == "RNXT":
        model = models.resnext101_64x4d(weights = "ResNeXt101_64X4D_Weights.IMAGENET1K_V1").to(device)
        modified_model = resnet.resnext101_64x4d(weights = "ResNeXt101_64X4D_Weights.IMAGENET1K_V1").to(device)
        model_rand = randomize_CNN_model(models.resnext101_64x4d(weights = "ResNeXt101_64X4D_Weights.IMAGENET1K_V1").to(device))
        modified_model_rand = randomize_CNN_model(resnet.resnext101_64x4d(weights = "ResNeXt101_64X4D_Weights.IMAGENET1K_V1").to(device))
        img_hw = 224
        num_patches = 0
        batch_size = 25
        model_list = [model, modified_model, model_rand, modified_model_rand]
    elif args.model == "VIT16":
        model = vit_base_patch16_224(pretrained=True).to(device)
        model_lrp = vit_base_patch16_224_LRP(pretrained=True).to(device)
        model_rand = randomize_VIT_model(vit_base_patch16_224(pretrained=True).to(device))
        model_lrp_rand = randomize_VIT_model(vit_base_patch16_224_LRP(pretrained=True).to(device))
        num_patches = 14
        batch_size = 25
        model_list = [model, model_lrp, model_rand, model_lrp_rand]
    elif args.model == "VIT32":
        model = vit_base_patch32_224(pretrained=True).to(device)
        model_lrp = vit_base_patch32_224_LRP(pretrained=True).to(device)
        model_rand = randomize_VIT_model(vit_base_patch32_224(pretrained=True).to(device))
        model_lrp_rand = randomize_VIT_model(vit_base_patch32_224_LRP(pretrained=True).to(device))
        num_patches = 7
        batch_size = 50
        model_list = [model, model_lrp, model_rand, model_lrp_rand]
    elif args.model == "CLIP16":
        clipmodel, preprocess = clip.load("ViT-B/16", device=device)
        mm_lrp_clipmodel, _ = mm_clip.load("ViT-B/16", device=device, jit=False)
        surgery_model, _ = surgery_clip.load("CS-ViT-B/16", device=device)
        m2ib_model = ClipWrapper(clipmodel).to(device)

        clipmodel_rand = randomize_CLIP_model(clip.load("ViT-B/16", device=device)[0])
        mm_lrp_clipmodel_rand = randomize_CLIP_model(mm_clip.load("ViT-B/16", device=device, jit=False)[0])
        surgery_model_rand = randomize_CLIP_model(surgery_clip.load("CS-ViT-B/16", device=device)[0])
        m2ib_model_rand = ClipWrapper(clipmodel_rand).to(device)


        clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
        num_patches = 14
        batch_size = 25
        model_list = [clipmodel, mm_lrp_clipmodel, surgery_model, m2ib_model, clip_tokenizer, clipmodel_rand, mm_lrp_clipmodel_rand, surgery_model_rand, m2ib_model_rand]
    elif args.model == "CLIP32":
        clipmodel, preprocess = clip.load("ViT-B/32", device=device)
        mm_lrp_clipmodel, _ = mm_clip.load("ViT-B/32", device=device, jit=False)
        surgery_model, _ = surgery_clip.load("CS-ViT-B/32", device=device)
        m2ib_model = ClipWrapper(clipmodel).to(device)

        clipmodel_rand = randomize_CLIP_model(clip.load("ViT-B/32", device=device)[0])
        mm_lrp_clipmodel_rand = randomize_CLIP_model(mm_clip.load("ViT-B/32", device=device, jit=False)[0])
        surgery_model_rand = randomize_CLIP_model(surgery_clip.load("CS-ViT-B/32", device=device)[0])
        m2ib_model_rand = ClipWrapper(clipmodel_rand).to(device)

        clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        num_patches = 7
        batch_size = 50
        model_list = [clipmodel, mm_lrp_clipmodel, surgery_model, m2ib_model, clip_tokenizer, clipmodel_rand, mm_lrp_clipmodel_rand, surgery_model_rand, m2ib_model_rand]

    # set up transforms
    if "R" in args.model:
        # CNN normalize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif "VIT" in args.model:
        # VIT normalize
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif "CLIP" in args.model:
        # CLIP_normalize
        normalize = preprocess

    transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])


    # get clip embeddings
    if "CLIP" in args.model:
        clip_labels = [f"a photo of a {label}" for label in class_list]
        all_text_processed = clip.tokenize(clip_labels).to(device)

        all_classes_embedding = [clipmodel.encode_text(text.unsqueeze(0)).detach().cpu() for text in all_text_processed]
        all_classes_embedding = torch.nn.functional.normalize(torch.from_numpy(np.array(all_classes_embedding)), dim=-1).to(device)

        all_classes_embedding_rand = [clipmodel_rand.encode_text(text.unsqueeze(0)).detach().cpu() for text in all_text_processed]
        all_classes_embedding_rand = torch.nn.functional.normalize(torch.from_numpy(np.array(all_classes_embedding_rand)), dim=-1).to(device)
    else:
        all_classes_embedding = None
        all_classes_embedding_rand = None

    testing_dict = {
        "models": model_list,
        "imagenet_dataset": args.dataset_path,
        "transform" : transform,
        "normalize": normalize,
        "embeddings": [all_classes_embedding, all_classes_embedding_rand],
        "img_hw": img_hw,
        "num_patches": num_patches,
        "batch_size": batch_size,
        "attr_func": args.attr_func,
        "model_name": args.model,
        "image_count": args.image_count,
        "device": device
    }

    # call the test function
    evaluate_sanity(testing_dict)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--image_count',
                        type = int, default = 1000,
                        help='How many images to test with.')
    parser.add_argument('--model',
                        type = str,
                        default = "R101",
                        help='Classifier to use: R101, RNXT, VIT32, VIT16, CLIP32, CLIP16')
    parser.add_argument('--attr_func',
                        type = str,
                        default = "IG",
                        help="attr to use: \
                            R101 or RNXT: {grad, inp_x_grad, ig, lig, idg, gig, agi, sg, xrai, gc, gbp, ggc, gs, lime, fa, occ}, \
                            VIT32 or VIT16: {attn, grad, n_rollout, rollout, t_attn, bi_attn, t_attr, VIT_CX, TIS, InFlow, MDA}, \
                            CLIP32 or CLIP16: {eclip, eclip_wo, game, maskclip, rollout, selfattn, surgery, m2ib, lrp} \
                        ")
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--dataset_path',
            type = str, default = "../../../ImageNet",
            help = 'The path to your dataset input')
    
    args, unparsed = parser.parse_known_args()
    
    main(args)