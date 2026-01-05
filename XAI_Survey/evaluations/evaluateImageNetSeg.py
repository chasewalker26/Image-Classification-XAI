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
from tqdm import tqdm

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
from util.attribution_methods import MASCalibrate
from util.attribution_methods.VIT_LRP.ViT_explanation_generator import Baselines, LRP
from util.attribution_methods.ViT_CX.ViT_CX import ViT_CX
from util.attribution_methods.TIS import TIS
from util.attribution_methods import MDAFunctions

from util.attribution_methods.CLIP.generate_emap import imgprocess_keepsize, mm_interpret, \
        clip_encode_dense, grad_eclip, mask_clip, compute_rollout_attention, \
        clip_surgery_map, m2ib_clip_map,  clip_lrp

# evaluation metrics
from util.test_methods import MASTestFunctions as MAS

# imagenet seg imports
from utils import render
from utils.saver import Saver
from utils.iou import IoU as IoU_metric
from skimage.segmentation import slic
from skimage.util import img_as_float
from data.Imagenet import Imagenet_Segmentation
from torch.utils.data import DataLoader, Subset

# RUN WITH
# python3 imagenet_seg_eval.py --method Calibrate_Best_Possible --gpu 3 --model VIT_base_32 --imagenet-seg-path ../../../gtsegs_ijcv.mat

cls = ['airplane',
       'bicycle',
       'bird',
       'boat',
       'bottle',
       'bus',
       'car',
       'cat',
       'chair',
       'cow',
       'dining table',
       'dog',
       'horse',
       'motobike',
       'person',
       'potted plant',
       'sheep',
       'sofa',
       'train',
       'tv'
       ]

with open('../../util/class_maps/ImageNet/imagenet_classes.txt') as f:
    class_list = [line.strip() for line in f.readlines()]

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

def get_CNN_attr(input_tensor, trans_img, target_class, testing_dict):
    model = testing_dict["models"][0]
    modified_model = testing_dict["models"][1]
    batch_size = testing_dict["batch_size"]
    img_hw = testing_dict["img_hw"]
    device = testing_dict["device"]
    attr_function = testing_dict["attr_func"]
    steps = 50
    baseline = 0

    resize = transforms.Resize((img_hw, img_hw), antialias = True)

    num_patches = 14
    downsize = transforms.Resize((num_patches, num_patches), antialias = True, interpolation=transforms.InterpolationMode.NEAREST_EXACT)
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
        agi_img = trans_img.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        mean = testing_dict["normalize"].mean 
        std = testing_dict["normalize"].std
        norm_layer = AGI.Normalize(mean, std)
        agi_model = torch.nn.Sequential(norm_layer, model).to(device)        
        selected_ids = range(0, 999, int(1000 / topk))
        _, _, agi = AGI.test(agi_model, device, agi_img, epsilon, topk, selected_ids, max_iter)
        percentile = 80
        upperbound = 99
        hm = agi

        try:
            hm = np.mean(hm, axis=0)
            q = np.percentile(hm, percentile)
            u = np.percentile(hm, upperbound)
            hm[hm<q] = q
            hm[hm>u] = u
            hm = (hm-q)/(u-q)
            saliency_map = torch.from_numpy(np.reshape(hm, (1, img_hw, img_hw)))
        except:
            saliency_map =  torch.rand((1, 1, 224, 224))

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
        img_lime = trans_img.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32) 
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

    saliency_map = torch.abs(torch.sum(saliency_map, dim = 0))
    Res = saliency_map.reshape(1, 1, img_hw, img_hw)
    Res = (Res - Res.min()) / (Res.max() - Res.min())
    ret = Res.mean()
    
    return Res, ret

def get_VIT_attr(input_tensor, trans_img, target_class, testing_dict, labels):
    model = testing_dict["models"][0]
    lrp_model = testing_dict["models"][1]
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
        saliency_map = torch.abs(torch.sum(saliency_map, dim = -1))
    elif attr_function == "TIS":
        saliency_method = TIS(model, batch_size=64)
        saliency_map = saliency_method(input_tensor.to(device), class_idx=target_class).cpu()
        saliency_map = resize(saliency_map.reshape((-1, saliency_map.shape[0], saliency_map.shape[1])).detach()).permute(1, 2, 0)
    elif attr_function == "InFlow":
        saliency_map, _ = explainer.generate_RAVE(input_tensor.to(device), target_class, option='b', device = device)
        saliency_map = resize(saliency_map.cpu().detach()).permute(1, 2, 0)
    elif attr_function == "MDA_sparse" or attr_function == "MDA":
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

        mda, _, _ = MDAFunctions.MDA(trans_img.cpu(), input_tensor.cpu(), bi_attn, num_patches ** 2, blur, model, device, img_hw, max_batch_size = 5)
        saliency_map = torch.from_numpy(mda)
        saliency_map = torch.abs(torch.sum(saliency_map, dim=-1))
    elif attr_function == "MDA_dense":
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

        segment_img = np.transpose(trans_img.squeeze().detach().cpu().numpy(), (1, 2, 0))
        segment_img = img_as_float(segment_img)
        segments = torch.tensor(slic(segment_img, n_segments=num_patches**2, compactness=10000, start_label=0), dtype = int)
        
        upsize = transforms.Resize((img_hw, img_hw), transforms.InterpolationMode.NEAREST_EXACT, antialias = True)
        downsize = transforms.Resize(num_patches, antialias = True)
        
        bi_attn, _ = explainer.bidirectional(input_tensor.to(device), target_class, device=device)
        reference_attr = resize(bi_attn.cpu().detach()).permute(1, 2, 0).cpu().numpy() * np.ones((img_hw, img_hw, 3))
        saliency_map_segmented = upsize(downsize(torch.tensor(reference_attr, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0))

        _, _, order_a, MR_ins = MASCalibrate.find_best_segment_reverse_informed(input_tensor.cpu(), saliency_map_segmented, segments, blur, num_patches**2, type = 1, model = model, device = device, img_hw = 224, max_batch_size = 25)
        end_index = np.where(MR_ins >= 0.9)[0][0]
        _, Res = MASCalibrate.find_deletion_from_insertion_informed_ultimate(input_tensor.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, num_patches**2, model, device, 224, max_batch_size = 5, kappa = -1)
        
        upsize = transforms.Resize(224, antialias=True)
        small_side = int(np.ceil(np.sqrt(num_patches**2)))
        downsize = transforms.Resize(small_side, antialias=True)
        Res = upsize(downsize(Res.permute((2, 0, 1)))).permute((1, 2, 0)).mean(dim=-1).reshape(1, 1, 224, 224)

    else:
        print("Model-attribution mismatch, please use --help.")
        exit()

    if "MDA_dense" not in attr_function:
        Res = saliency_map.reshape(1, 1, img_hw, img_hw)
        Res = (Res - Res.min()) / (Res.max() - Res.min())
        ret = Res.mean()
    # evaluate all possible thresholds to pick the best performing
    else:
        Res = (Res - Res.min()) / (Res.max() - Res.min())
        # vary kappa
        mag_vals = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

        Res = Res / Res.mean() * 0.5

        acc_array = np.zeros(len(mag_vals))
        IoU_array = np.zeros(len(mag_vals))
        AP_array = np.zeros(len(mag_vals))
        f1_array = np.zeros(len(mag_vals))
        for i in range(len(mag_vals)):
            Res_mean_1 = Res.gt(mag_vals[i]).type(Res.type())
            Res_mean_0 = Res.le(mag_vals[i]).type(Res.type())
            output = torch.cat((Res_mean_0, Res_mean_1), 1)
            correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
            inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)

            AP_array[i] = np.nan_to_num(get_ap_scores(output, labels))[0]
            acc_array[i] = np.float64(1.0) * correct / (np.spacing(1, dtype=np.float64) + labeled).squeeze()
            IoU_array[i] = np.mean(np.float64(1.0) * inter / (np.spacing(1, dtype=np.float64) + union))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f1_array[i] = np.mean(np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0])))

        selected_index = np.argmax(IoU_array)

        ret = mag_vals[selected_index]

    return Res, ret

def get_CLIP_attr(trans_img, target_class, testing_dict):
    clipmodel = testing_dict["models"][0]
    mm_lrp_clipmodel = testing_dict["models"][1]
    surgery_model = testing_dict["models"][2]
    m2ib_model = testing_dict["models"][3]
    clip_tokenizer = testing_dict["models"][4]
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
    # elif attr_function == "MDA":
    #     img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
    #     klen = 31
    #     ksig = 31
    #     kern = MAS.gkern(klen, ksig)
    #     blur = lambda x: torch.nn.functional.conv2d(x, kern, padding = klen // 2)

    #     _, blur_perc = get_CLIP_pred(blur(img_clipreprocess.cpu()).to(device), clipmodel, testing_dict["embeddings"])
    #     while blur_perc > 1:
    #         klen += 4
    #         ksig += 4
    #         kern = MAS.gkern(klen, ksig)
    #         blur = lambda x: torch.nn.functional.conv2d(x, kern, padding = klen // 2)
    #         _, blur_perc = get_CLIP_pred(blur(img_clipreprocess.cpu()).to(device), clipmodel, testing_dict["embeddings"])

    #     emap = [grad_eclip(c, q_out, k_out, v, att_output, map_size, withksim=True) for c in cosines]
    #     emap = torch.stack(emap, dim=0).sum(0)  
    #     emap = resize(emap.unsqueeze(0))[0].squeeze().reshape((img_hw, img_hw, -1)).detach().cpu().numpy().reshape(1, img_hw, img_hw) * np.ones((3, img_hw, img_hw)) 

    #     CLIP_test_info = {
    #         "input": img_clipreprocess,
    #         "embeddings": testing_dict["embeddings"],
    #         "prediction_function": get_CLIP_pred
    #     }

    #     mda, _, _ = MDAFunctions.MDA(trans_img, img_clipreprocess.cpu(), emap, num_patches ** 2, blur, clipmodel, device, img_hw, max_batch_size = 5, CLIP_test_info = CLIP_test_info)
    #     saliency_map = torch.from_numpy(mda)
    else:
        print("Model-attribution mismatch, please use --help.")
        exit()

    if attr_function != "MDA":
        saliency_map = resize(emap.unsqueeze(0))[0].squeeze().reshape((img_hw, img_hw, -1))

    Res = saliency_map.reshape(1, 1, img_hw, img_hw)
    Res = (Res - Res.min()) / (Res.max() - Res.min())
    ret = Res.mean()

    return Res, ret

# Res is an attribution, ret is the attr thresh, labels is segmentation mask
def eval_batch(Res, ret, labels):
    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1-Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    # TEST
    pred = Res.clamp(min=0) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    output = torch.cat((Res_0, Res_1), 1)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)

    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    ap = np.nan_to_num(get_ap_scores(output, labels))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target

def evaluate_imagenet_seg(testing_dict):
    device = testing_dict["device"]

    iterator = tqdm(testing_dict["segmentation_dataloader"])
    total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    total_ap, total_f1 = [], []
    predictions, targets = [], []
    for batch_idx, (image, labels) in enumerate(iterator):
        trans_img = image.to(device).squeeze()
        labels = labels.to(device)

        # Get the class and prediciton for the image
        if "CLIP" in testing_dict["model_name"]:
            input_tensor = testing_dict["normalize"](transforms.functional.to_pil_image(trans_img))
            input_tensor = torch.unsqueeze(input_tensor, 0).to(testing_dict["device"])
            target_class, original_pred = get_CLIP_pred(input_tensor, testing_dict["models"][0], testing_dict["embeddings"])
        else:
            input_tensor = testing_dict["normalize"](trans_img)
            input_tensor = torch.unsqueeze(input_tensor, 0).to(testing_dict["device"])
            target_class, original_pred = get_classifier_pred(input_tensor, testing_dict["models"][0], testing_dict["device"])
                        
        if "R" in testing_dict["model_name"]:
            Res, ret = get_CNN_attr(input_tensor, trans_img, target_class, testing_dict)
        elif "VIT" in testing_dict["model_name"]:
            Res, ret = get_VIT_attr(input_tensor, trans_img, target_class, testing_dict, labels)
        elif "CLIP" in testing_dict["model_name"]:
            Res, ret = get_CLIP_attr(trans_img, target_class, testing_dict)
        
        correct, labeled, inter, union, ap, f1, pred, target = eval_batch(Res, ret, labels)

        if type(correct) is int:
            continue

        predictions.append(pred)
        targets.append(target)

        total_correct += correct.astype('int64')
        total_label += labeled.astype('int64')
        total_inter += inter.astype('int64')
        total_union += union.astype('int64')
        total_ap += [ap]
        total_f1 += [f1]
        pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
        IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
        mIoU = IoU.mean()
        mAp = np.mean(total_ap)
        mF1 = np.mean(total_f1)
        iterator.set_description(testing_dict["model_name"] + ' pixAcc ' + testing_dict['attr_func'] + ': %.4f, mIoU: %.4f, mAP: %.4f, mF1: %.4f' % (pixAcc, mIoU, mAp, mF1))

    # make the test folder if it doesn't exist
    folder = "seg_test_results/" + testing_dict["model_name"] + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_name = testing_dict["attr_func"] + "_" + str(testing_dict["image_count"]) + "_images"

    txtfile = os.path.join(folder, file_name)
    fh = open(txtfile, 'w')
    fh.write("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
    fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
    fh.write("Mean AP over %d classes: %.4f\n" % (2, mAp))
    fh.write("Mean F1 over %d classes: %.4f\n" % (2, mF1))
    fh.close()

    return

def main(args):
    img_hw = 224
    device = 'cuda:' + str(args.cuda_num) if torch.cuda.is_available() else 'cpu'

    # set up models 
    if args.model == "R101":
        model = models.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2").to(device)
        modified_model = resnet.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2").to(device)
        img_hw = 224
        num_patches = 0
        batch_size = 50
        model_list = [model, modified_model]
    elif args.model == "RNXT":
        model = models.resnext101_64x4d(weights = "ResNeXt101_64X4D_Weights.IMAGENET1K_V1").to(device)
        modified_model = resnet.resnext101_64x4d(weights = "ResNeXt101_64X4D_Weights.IMAGENET1K_V1").to(device)
        img_hw = 224
        num_patches = 0
        batch_size = 25
        model_list = [model, modified_model]
    elif args.model == "VIT16":
        model = vit_base_patch16_224(pretrained=True).to(device)
        model_lrp = vit_base_patch16_224_LRP(pretrained=True).to(device)
        num_patches = 14
        batch_size = 25
        model_list = [model, model_lrp]
    elif args.model == "VIT32":
        model = vit_base_patch32_224(pretrained=True).to(device)
        model_lrp = vit_base_patch32_224_LRP(pretrained=True).to(device)
        num_patches = 7
        batch_size = 50
        model_list = [model, model_lrp]
    elif args.model == "CLIP16":
        clipmodel, preprocess = clip.load("ViT-B/16", device=device)
        mm_lrp_clipmodel, _ = mm_clip.load("ViT-B/16", device=device, jit=False)
        surgery_model, _ = surgery_clip.load("CS-ViT-B/16", device=device)
        m2ib_model = ClipWrapper(clipmodel).to(device)
        clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
        num_patches = 14
        batch_size = 25
        model_list = [clipmodel, mm_lrp_clipmodel, surgery_model, m2ib_model, clip_tokenizer]
    elif args.model == "CLIP32":
        clipmodel, preprocess = clip.load("ViT-B/32", device=device)
        mm_lrp_clipmodel, _ = mm_clip.load("ViT-B/32", device=device, jit=False)
        surgery_model, _ = surgery_clip.load("CS-ViT-B/32", device=device)
        m2ib_model = ClipWrapper(clipmodel).to(device)
        clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        num_patches = 7
        batch_size = 50
        model_list = [clipmodel, mm_lrp_clipmodel, surgery_model, m2ib_model, clip_tokenizer]

    # set up normalizations
    if "R" in args.model:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif "VIT" in args.model:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif "CLIP" in args.model:
        normalize = preprocess

    # get clip embeddings
    if "CLIP" in args.model:
        clip_labels = [f"a photo of a {label}" for label in class_list]
        all_text_processed = clip.tokenize(clip_labels).to(device)
        all_classes_embedding = [clipmodel.encode_text(text.unsqueeze(0)).detach().cpu() for text in all_text_processed]
        all_classes_embedding = torch.nn.functional.normalize(torch.from_numpy(np.array(all_classes_embedding)), dim=-1).to(device)
    else:
        all_classes_embedding = None

    # setup the segmentation dataset
    test_img_trans = transforms.Compose([
        transforms.Resize((224, 224), antialias = True),
        transforms.ToTensor()
    ])
    test_lbl_trans = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT, antialias = True)
    ])
    ds = Imagenet_Segmentation(args.dataset_path, transform=test_img_trans, target_transform=test_lbl_trans)
    if args.image_count != -1:
        subset_indices = list(range(0, args.image_count)) 
    else: 
        subset_indices = list(range(0, len(ds))) 
        args.image_count = len(ds)

    ds = Subset(ds, subset_indices)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    testing_dict = {
        "models": model_list,
        "segmentation_dataloader": dl,
        "normalize": normalize,
        "embeddings": all_classes_embedding,
        "img_hw": img_hw,
        "image_count": args.image_count,
        "num_patches": num_patches,
        "batch_size": batch_size,
        "attr_func": args.attr_func,
        "model_name": args.model,
        "device": device
    }

    # call the test function
    evaluate_imagenet_seg(testing_dict)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--model',
                        type = str,
                        default = "R101",
                        help='Classifier to use: R101, RNXT, VIT32, VIT16, CLIP32, CLIP16')
    parser.add_argument('--image_count',
                    type = int, default = -1,
                    help='How many images to test with. -1 indicates all images')
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
            type = str, default = "../../../gtsegs_ijcv.mat",
            help = 'The path to your dataset input')
    
    args, unparsed = parser.parse_known_args()
    
    main(args)