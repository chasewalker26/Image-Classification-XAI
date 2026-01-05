import numpy as np
import torch
import torchvision.transforms as transforms
from numpy import *
import argparse
import os
import csv
import warnings
import time
from PIL import Image
from collections import Counter
from transformers import CLIPTokenizerFast
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

from util.visualization import attr_to_subplot

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
from util.test_methods import AICTestFunctions as PIC
from util.test_methods import MonotonicityTest as MONO
from util.test_methods import PosNegPertFunctions as PNP


blue_cmap = LinearSegmentedColormap.from_list(
    'custom_blue_light',
    [
        (0.00, (1.00, 1.00, 1.00)),   # white
        (0.25, (0.35, 0.40, 1.00)),   # very light blue
        (0.50, (0.25, 0.30, 1.00)),
        (0.75, (0.15, 0.20, 1.00)),
        (1.00, (0.00, 0.00, 1.00))    # full blue
    ],
    N=256
)

# heatmap
def heatmap_overlap_axs(img, attr, name, axs, cmap):
    attr_to_subplot(attr, name, axs, cmap = cmap, norm = 'absolute', blended_image = img, alpha = 1)
    return

with open('../../util/class_maps/ImageNet/ILSVRC2012_validation_ground_truth.txt') as f:
    gnd_truth = [int(line.strip()) for line in f.readlines()]
with open('../../util/class_maps/ImageNet/map_clsloc.txt') as f:
    class_map = [line.strip() for line in f.readlines()] 
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

def get_CNN_attr(input_tensor, trans_img, target_class, testing_dict, name, path):
    model = testing_dict["models"][0]
    modified_model = testing_dict["models"][1]
    batch_size = testing_dict["batch_size"]
    img_hw = testing_dict["img_hw"]
    device = testing_dict["device"]
    steps = 50
    baseline = 0

    resize = transforms.Resize((img_hw, img_hw), antialias = True)

    num_patches = 14
    downsize = transforms.Resize((num_patches, num_patches), interpolation=transforms.InterpolationMode.NEAREST_EXACT)
    patch_ids = torch.arange(num_patches ** 2).reshape((num_patches, num_patches))
    patch_mask = patch_ids.repeat_interleave(int(img_hw / num_patches), dim=0).repeat_interleave(int(img_hw / num_patches), dim=1).to(device)

    input_tensor.requires_grad = True
    grad, _ = attr.getGradientsParallel(input_tensor.to(device), model, target_class)
    input_tensor.requires_grad = False

    ig = attr.IG(input_tensor, model, steps, batch_size, 1, baseline, device, target_class)
    
    lig = attr.IG(input_tensor, model, steps, batch_size, .9, baseline, device, target_class)

    idg = attr.IDG(input_tensor, model, steps, batch_size, baseline, device, target_class)

    call_model_args = {'class_idx_str': target_class.item()}
    guided_ig = GIG_Builder.GuidedIG()
    baseline = torch.zeros_like(input_tensor.cpu())
    gig = guided_ig.GetMask(input_tensor.cpu(), model, device, GIG_Builder.call_model_function, call_model_args, x_baseline=baseline, x_steps=steps, max_dist=1.0, fraction=0.5).squeeze()

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
    agi = np.reshape(hm, (img_hw, img_hw, 1))

    sg = attr.smoothGrad("IG", input_tensor, model, 50, baseline, target_class, device)

    xrai_object = XRAI.XRAI()
    xrai = xrai_object.GetMask(input_tensor.squeeze().permute(1, 2, 0).cpu(), base_attribution = ig.permute(1, 2, 0).cpu().detach()).reshape((224, 224, 1))

    layer = model.layer4
    layer_gc = LayerGradCam(model, layer)
    input_tensor.requires_grad = True
    gc = layer_gc.attribute(input_tensor.to(device), target_class, relu_attributions=True)
    input_tensor.requires_grad = False
    gc = resize(gc.squeeze().reshape((1, 7, 7))).reshape((1, img_hw, img_hw)).cpu() * torch.ones((3, img_hw, img_hw))
    
    guided_bp = GuidedBackprop(modified_model)
    input_tensor.requires_grad = True
    gbp = guided_bp.attribute(input_tensor.to(device), target = target_class).squeeze()
    input_tensor.requires_grad = False

    guided_gc = GuidedGradCam(modified_model, modified_model.layer4)
    input_tensor.requires_grad = True
    ggc = guided_gc.attribute(input_tensor.to(device), target_class).squeeze()
    input_tensor.requires_grad = False

    baselines = torch.randn(1, 3, 224, 224)
    gradient_shap = GradientShap(model)
    gs = gradient_shap.attribute(input_tensor.to(device), baselines.to(device), target = target_class).squeeze()
    
    img_lime = trans_img.permute(1, 2, 0).numpy().astype(np.float32) 
    lime = limeAttr.get_lime_attr(img_lime, model, device)
    
    ablator = FeatureAblation(model)
    FA = resize(downsize(ablator.attribute(input_tensor.to(device), target=target_class, feature_mask=patch_mask))).squeeze()

    ablator = Occlusion(model)
    OCC = resize(downsize(ablator.attribute(input_tensor.to(device), target=target_class, sliding_window_shapes=(3,64,64), strides=32))).squeeze()

    grad_print = np.transpose(grad.squeeze().detach().cpu().numpy(), (1,2,0))
    ig_print = np.transpose(ig.squeeze().detach().cpu().numpy(), (1,2,0))
    lig_print = np.transpose(lig.squeeze().detach().cpu().numpy(), (1,2,0))
    idg_print = np.transpose(idg.squeeze().detach().cpu().numpy(), (1,2,0))
    sg_print =  np.transpose(sg.squeeze().detach().cpu().numpy(), (1,2,0))
    gc_print = np.transpose(gc.squeeze().detach().cpu().numpy(), (1,2,0))
    ggc_print = np.transpose(ggc.squeeze().detach().cpu().numpy(), (1,2,0))
    gbp_print = np.transpose(gbp.squeeze().detach().cpu().numpy(), (1,2,0))
    gig_print = np.transpose(gig.squeeze().detach().cpu().numpy(), (1, 2, 0))
    agi_print = agi
    lime_print = np.transpose(lime.squeeze().detach().cpu().numpy(), (1, 2, 0))
    FA_print = np.transpose(FA.squeeze().detach().cpu().numpy(), (1, 2, 0))
    OCC_print = np.transpose(OCC.squeeze().detach().cpu().numpy(), (1, 2, 0))
    gs_print = np.transpose(gs.squeeze().detach().cpu().numpy(), (1,2,0))
    xrai_print = xrai

    plt.rcParams.update({'font.size': 30})
    fig, axs = plt.subplots(1, 16, figsize = (48, 3))
    attr_to_subplot(trans_img, "Input", axs[0], original_image = True)
    heatmap_overlap_axs(trans_img, grad_print, 'Grad', axs[1], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, gbp_print, 'GBP', axs[2], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, gc_print, 'GC', axs[3], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, ig_print, 'IG', axs[4], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, lig_print, 'LIG', axs[5], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, gig_print, 'GIG', axs[6], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, agi_print, 'AGI', axs[7], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, idg_print, 'IDG', axs[8], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, FA_print, 'FA', axs[9], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, OCC_print, 'OCC', axs[10], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, gs_print, 'GS', axs[11], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, lime_print, 'LIME', axs[12], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, sg_print, 'SG', axs[13], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, xrai_print, 'XRAI', axs[14], cmap = blue_cmap)
    heatmap_overlap_axs(trans_img, ggc_print, 'GGC', axs[15], cmap = blue_cmap)
    axs[0].set_ylabel(name)
    plt.subplots_adjust(wspace = 0.05)
    fig.savefig(path + '.png', dpi=100, bbox_inches='tight')

    return

def get_VIT_attr(input_tensor, trans_img, target_class, testing_dict, name, path):
    model = testing_dict["models"][0]
    lrp_model = testing_dict["models"][1]
    num_patches = testing_dict["num_patches"]
    img_hw = testing_dict["img_hw"]
    device = testing_dict["device"]
    batch_size = testing_dict["batch_size"]

    resize = transforms.Resize((img_hw, img_hw), antialias = True)
    resize_square = transforms.Resize((img_hw, img_hw), antialias = True, interpolation=transforms.InterpolationMode.NEAREST_EXACT)

    patch_ids = torch.arange(num_patches ** 2).reshape((num_patches, num_patches))
    patch_mask = patch_ids.repeat_interleave(int(224 / num_patches), dim=0).repeat_interleave(int(224 / num_patches), dim=1)

    explainer = Baselines(model)
    LRP_explainer = LRP(lrp_model)


    saliency_map = explainer.generate_raw_attn(input_tensor.to(device), device)
    attn = resize(saliency_map.cpu().detach()).permute(1, 2, 0).numpy()

    saliency_map = explainer.generate_grad(input_tensor.to(device), target_class, device)
    attn_grad = resize(saliency_map.cpu().detach()).permute(1, 2, 0).numpy()

    saliency_map, _, _ = explainer.generate_rollout(input_tensor.to(device))
    rollout = resize(saliency_map.cpu().detach()).permute(1, 2, 0).numpy()

    _, _, saliency_map, _, _ = explainer.generate_transition_attention_maps(input_tensor.to(device), target_class, start_layer = 0, device = device)
    t_attn = resize(saliency_map.cpu().detach()).permute(1, 2, 0).numpy()

    IG = explainer.IG(input_tensor.to(device), target_class)
    IG = resize(IG.cpu().detach()).permute(1, 2, 0).numpy()

    saliency_map, _ = explainer.bidirectional(input_tensor.to(device), target_class, device = device)
    bi_attn = resize(saliency_map.cpu().detach()).permute(1, 2, 0).numpy()

    saliency_map = LRP_explainer.generate_LRP(input_tensor.to(device), target_class, method="transformer_attribution", start_layer = 0, device = device)
    T_attr = resize(saliency_map.cpu().detach()).permute(1, 2, 0).numpy()

    target_layer = model.blocks[-1].norm1
    result, _ = ViT_CX(model, input_tensor, target_layer, gpu_batch=1, device = device)
    saliency_map = (result.reshape((img_hw, img_hw, 1)) * torch.ones((img_hw, img_hw, 3)))
    vit_cx = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    vit_cx = vit_cx.cpu().detach().numpy()

    saliency_method = TIS(model, batch_size=64)
    saliency_map = saliency_method(input_tensor.to(device), class_idx=target_class).cpu()
    tis = resize(saliency_map.reshape((-1, saliency_map.shape[0], saliency_map.shape[1])).detach()).permute(1, 2, 0).numpy()

    saliency_map, _ = explainer.generate_RAVE(input_tensor.to(device), target_class, option='b', device = device)
    inflow = resize(saliency_map.cpu().detach()).permute(1, 2, 0).numpy()

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

    plt.rcParams.update({'font.size': 30})
    fig, axs = plt.subplots(1, 12, figsize = (36, 3))
    attr_to_subplot(trans_img, "Input", axs[0], original_image = True)
    heatmap_overlap_axs(trans_img, attn, 'Attn', axs[1], cmap = 'jet')
    heatmap_overlap_axs(trans_img, attn_grad, 'Grad', axs[2], cmap = 'jet')
    heatmap_overlap_axs(trans_img, rollout, 'Rollout', axs[3], cmap = 'jet')
    heatmap_overlap_axs(trans_img, IG, 'IG', axs[4], cmap = 'jet')
    heatmap_overlap_axs(trans_img, t_attn, 'T-Attn', axs[5], cmap = 'jet')
    heatmap_overlap_axs(trans_img, bi_attn, 'Bi-Attn', axs[6], cmap = 'jet')
    heatmap_overlap_axs(trans_img, vit_cx, 'ViT-CX', axs[7], cmap = 'jet')
    heatmap_overlap_axs(trans_img, tis, 'TIS', axs[8], cmap = 'jet')
    heatmap_overlap_axs(trans_img, T_attr, 'T-Attr', axs[9], cmap = 'jet')
    heatmap_overlap_axs(trans_img, inflow, 'InFlow', axs[10], cmap = 'jet')
    heatmap_overlap_axs(trans_img, mda, 'MDA', axs[11], cmap = 'jet')
    axs[0].set_ylabel(name)
    plt.subplots_adjust(wspace = 0.05)
    fig.savefig(path + '.png', dpi=100, bbox_inches='tight')

    return 

def get_CLIP_attr(trans_img, target_class, testing_dict, name, path):
    clipmodel = testing_dict["models"][0]
    mm_lrp_clipmodel = testing_dict["models"][1]
    surgery_model = testing_dict["models"][2]
    m2ib_model = testing_dict["models"][3]
    clip_tokenizer = testing_dict["models"][4]
    preprocess = testing_dict["normalize"]
    device = testing_dict["device"]
    img_hw = testing_dict["img_hw"]
    num_patches = testing_dict["num_patches"]

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


    emap = [grad_eclip(c, q_out, k_out, v, att_output, map_size) for c in cosines]
    eclip = torch.stack(emap, dim=0).sum(0)
    eclip = resize(eclip.unsqueeze(0))[0].squeeze().reshape((224, 224, -1)).detach().cpu().numpy()

    emap = [grad_eclip(c, q_out, k_out, v, att_output, map_size, withgrad = False) for c in cosines]
    eclip_nograd = torch.stack(emap, dim=0).sum(0) 
    eclip_nograd = resize(eclip_nograd.unsqueeze(0))[0].squeeze().reshape((224, 224, -1)).detach().cpu().numpy()

    img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
    text_tokenized = mm_clip.tokenize(txts).to(device)
    emap = mm_interpret(model=mm_lrp_clipmodel, image=img_clipreprocess, texts=text_tokenized, device=device)    
    game = emap.sum(0)
    game = resize(game.unsqueeze(0))[0].squeeze().reshape((224, 224, -1)).detach().cpu().numpy()

    emap = mask_clip(txt_embedding.T, v_final, k_out, map_size)
    maskclip = emap.sum(0)
    maskclip = resize(maskclip.unsqueeze(0))[0].squeeze().reshape((224, 224, -1)).detach().cpu().numpy()

    img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
    text_tokenized = mm_clip.tokenize(txts).to(device)
    attentions = mm_interpret(model=mm_lrp_clipmodel, image=img_clipreprocess, texts=text_tokenized, device=device, rollout=True)      
    rollout = compute_rollout_attention(attentions)[0]
    rollout = resize(rollout.unsqueeze(0))[0].squeeze().reshape((224, 224, -1)).detach().cpu().numpy()

    selfattn = attn[0,:1,1:].detach().reshape(*map_size)
    selfattn = resize(selfattn.unsqueeze(0))[0].squeeze().reshape((224, 224, -1)).detach().cpu().numpy()

    img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
    all_texts = ['airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform', 'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']
    all_texts = txts + all_texts
    surgery = clip_surgery_map(model=surgery_model, image=img_clipreprocess, texts=all_texts, device=device)[0,:,:,0]
    surgery = resize(surgery.unsqueeze(0))[0].squeeze().reshape((224, 224, -1)).detach().cpu().numpy()

    img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
    m2ib = m2ib_clip_map(model=m2ib_model, clip_tokenizer=clip_tokenizer, image=img_clipreprocess, texts=txts[0], device=device)
    m2ib = resize(torch.tensor(m2ib).unsqueeze(0))[0].squeeze().reshape((224, 224, -1)).detach().cpu().numpy()

    img_clipreprocess = preprocess(img).to(device).unsqueeze(0)
    _, hm = clip_lrp(img_clipreprocess, lrp_clip.tokenize(txts[0]).to(device), mm_lrp_clipmodel, device)
    lrp = hm.reshape((-1, num_patches, num_patches))
    lrp = resize(lrp.unsqueeze(0))[0].squeeze().reshape((224, 224, -1)).detach().cpu().numpy()


    plt.rcParams.update({'font.size': 30 })
    fig, axs = plt.subplots(1, 10, figsize = (30, 3))
    attr_to_subplot(trans_img, "Input", axs[0], original_image = True)
    heatmap_overlap_axs(trans_img, eclip, 'ECLIP', axs[1], cmap = 'jet')
    heatmap_overlap_axs(trans_img, eclip_nograd, 'GradECLIP', axs[2], cmap = 'jet')
    heatmap_overlap_axs(trans_img, game, 'GAME', axs[3], cmap = 'jet')
    heatmap_overlap_axs(trans_img, lrp, 'T-Attr', axs[4], cmap = 'jet')
    heatmap_overlap_axs(trans_img, m2ib, 'M2iB', axs[5], cmap = 'jet')
    heatmap_overlap_axs(trans_img, maskclip, 'MASKCLIP', axs[6], cmap = 'jet')
    heatmap_overlap_axs(trans_img, rollout, 'Rollout', axs[7], cmap = 'jet')
    heatmap_overlap_axs(trans_img, selfattn, 'Attn', axs[8], cmap = 'jet')
    heatmap_overlap_axs(trans_img, surgery, 'Surgery', axs[9], cmap = 'jet')
    axs[0].set_ylabel(name)
    plt.subplots_adjust(wspace = 0.05)
    fig.savefig(path + '.png', dpi=100, bbox_inches='tight')

    return

def create_attr_figure(testing_dict):
    # initialize the blur kernel
    klen = 31
    ksig = 31
    kern = MAS.gkern(klen, ksig)
    blur = lambda x: torch.nn.functional.conv2d(x, kern, padding = klen // 2)

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + testing_dict["model_name"] + ".txt").astype(np.int64)

    with open('../../util/class_maps/ImageNet/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    num_classes = 1000
    images_per_class = int(np.ceil(testing_dict["image_count"] / num_classes))
    classes_used = [0] * num_classes

    # make the test folder if it doesn't exist
    folder = "qualitative_results/" + testing_dict["model_name"] + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    images_used = 0
    # look at test images in order from 1
    with tqdm(total = testing_dict["image_count"], desc = testing_dict["model_name"] + " Qualitative Eval Gen") as pbar:
        # look at validations starting from 1
        for image in sorted(os.listdir(testing_dict["imagenet_dataset"])):    
            if images_used == testing_dict["image_count"]:
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
                target_class, original_pred = get_CLIP_pred(input_tensor, testing_dict["models"][0], testing_dict["embeddings"])
                
                blur_class, blur_pred = get_CLIP_pred(blur(input_tensor.cpu()).to(testing_dict["device"]), testing_dict["models"][0], testing_dict["embeddings"])
                black_class, black_pred = get_CLIP_pred(torch.zeros_like(input_tensor).to(testing_dict["device"]), testing_dict["models"][0], testing_dict["embeddings"])
            else:
                input_tensor = testing_dict["normalize"](trans_img)
                input_tensor = torch.unsqueeze(input_tensor, 0).to(testing_dict["device"])
                target_class, original_pred = get_classifier_pred(input_tensor, testing_dict["models"][0], testing_dict["device"])

                blur_class, blur_pred = get_classifier_pred(blur(input_tensor.cpu()).to(testing_dict["device"]), testing_dict["models"][0], testing_dict["device"])
                black_class, black_pred = get_classifier_pred(torch.zeros_like(input_tensor).to(testing_dict["device"]), testing_dict["models"][0], testing_dict["device"])
            
            # check if the tests will function properly
            if blur_pred >= original_pred or black_pred >= original_pred or target_class == black_class or target_class == blur_class:
                continue

            # Track which classes have been used
            if classes_used[target_class] == images_per_class:
                continue
            else:
                classes_used[target_class] += 1
            
            name = classes[target_class]

            if "R" in testing_dict["model_name"]:
                get_CNN_attr(input_tensor, trans_img, target_class, testing_dict, name, folder + image)
            elif "VIT" in testing_dict["model_name"]:
                get_VIT_attr(input_tensor, trans_img, target_class, testing_dict, name, folder + image)
            elif "CLIP" in testing_dict["model_name"]:
                get_CLIP_attr(trans_img, target_class, testing_dict, name, folder + image)
                        
            images_used += 1
            pbar.update(1)
    return

def main(args):
    img_hw = 224
    device = 'cuda:' + str(args.cuda_num) if torch.cuda.is_available() else 'cpu'

    # set up models 
    if args.model == "R101":
        model = models.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2").to(device).eval()
        modified_model = resnet.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2").to(device).eval()
        img_hw = 224
        num_patches = 0
        batch_size = 50
        model_list = [model, modified_model]
    elif args.model == "R152":
        model = models.resnet101(weights = "ResNet152_Weights.IMAGENET1K_V2").to(device).eval()
        modified_model = resnet.resnet101(weights = "ResNet152_Weights.IMAGENET1K_V2").to(device).eval()
        img_hw = 224
        num_patches = 0
        batch_size = 50
        model_list = [model, modified_model]
    elif args.model == "RNXT":
        model = models.resnext101_64x4d(weights = "ResNeXt101_64X4D_Weights.IMAGENET1K_V1").to(device).eval()
        modified_model = resnet.resnext101_64x4d(weights = "ResNeXt101_64X4D_Weights.IMAGENET1K_V1").to(device).eval()
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
    else:
        all_classes_embedding = None

    testing_dict = {
        "models": model_list,
        "imagenet_dataset": args.dataset_path,
        "transform" : transform,
        "normalize": normalize,
        "embeddings": all_classes_embedding,
        "img_hw": img_hw,
        "num_patches": num_patches,
        "batch_size": batch_size,
        "model_name": args.model,
        "image_count": args.image_count,
        "device": device
    }

    # call the test function
    create_attr_figure(testing_dict)

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
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--dataset_path',
            type = str, default = "../../../ImageNet",
            help = 'The path to your dataset input')
    
    args, unparsed = parser.parse_known_args()
    
    main(args)