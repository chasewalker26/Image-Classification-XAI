from torchvision import models
from torchvision import transforms
from transformers import SwinForImageClassification, PvtForImageClassification, utils
from PIL import Image
import os, sys
import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import login

utils.logging.set_verbosity_error()  # Suppress standard warnings

os.sys.path.append(os.path.dirname(os.path.abspath('.')))

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from attribution_methods.VIT_LRP.ViT_new_timm import vit_tiny_patch16_224, vit_base_patch32_224, vit_base_patch16_224, vit_base_patch8_224, vit_large_patch16_224

def getClass(input_batch, model, device):
    with torch.no_grad():
        output = model(input_batch.to(device))

        if not isinstance(output, torch.Tensor):
            output = output.logits

        _, indices = torch.max(output, dim=1)  # shape: (B,)
    
    return indices

def main():
    login('your_hf_token')

    model_select = int(sys.argv[1])

    if (model_select >= 0 and model_select <= 7) or model_select >=13:
        transform_normalize = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    else:
        transform_normalize = transforms.Normalize(
            mean = [0.5, 0.5, 0.5],
            std = [0.5, 0.5, 0.5]
        )

    # img_hw determines how to transform input images for model needs
    if model_select == 0:
        model = models.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2")
        img_hw = 224
        model_name = "R101"
    elif model_select == 1:
        model = models.resnet152(weights = "ResNet152_Weights.IMAGENET1K_V2")
        img_hw = 224
        model_name = "R152"
    elif model_select == 2:
        model = models.inception_v3(weights = "Inception_V3_Weights.IMAGENET1K_V1")
        img_hw = 299
        model_name = "IV3"
    elif model_select == 3:
        model = models.vgg19(weights = "VGG19_Weights.IMAGENET1K_V1")
        img_hw = 224
        model_name = "VGG19"
    elif model_select == 4:
        model = models.convnext_large(weights = "ConvNeXt_Large_Weights.IMAGENET1K_V1")
        img_hw = 224
        model_name = "CONVNXT"
    elif model_select == 5:
        model = models.resnext101_64x4d(weights = "ResNeXt101_64X4D_Weights.IMAGENET1K_V1")
        img_hw = 224
        model_name = "RESNXT"
    elif model_select == 6:
        model = models.vit_b_16(weights = "ViT_B_16_Weights.IMAGENET1K_V1")
        img_hw = 224
        model_name = "VIT16"
    elif model_select == 7:
        model = models.vit_b_32(weights = "ViT_B_32_Weights.IMAGENET1K_V1")
        img_hw = 224
        model_name = "VIT32"
    elif model_select == 8:
        model = vit_tiny_patch16_224(pretrained=True)
        img_hw = 224
        model_name = "VIT_tiny_16"
    elif model_select == 9:
        model = vit_base_patch32_224(pretrained=True)
        img_hw = 224
        model_name = "VIT_base_32"
    elif model_select == 10:
        model = vit_base_patch16_224(pretrained=True)
        img_hw = 224
        model_name = "VIT_base_16"
    elif model_select == 11:
        model = vit_base_patch8_224(pretrained=True)
        img_hw = 224
        model_name = "VIT_base_8"
    elif model_select == 12:
        model = vit_large_patch16_224(pretrained=True)
        img_hw = 224
        model_name = "VIT_large_16"
    elif model_select == 13:
        model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        img_hw = 224
        model_name = "swin_tiny"
    elif model_select == 14:
        model = SwinForImageClassification.from_pretrained("microsoft/swin-small-patch4-window7-224")
        img_hw = 224
        model_name = "swin_small"
    elif model_select == 15:
        model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")
        img_hw = 224
        model_name = "swin_base"
    elif model_select == 16:
        model = PvtForImageClassification.from_pretrained("Xrenya/pvt-tiny-224")
        img_hw = 224
        model_name = "pvt_tiny"
    elif model_select == 17:
        model = PvtForImageClassification.from_pretrained("Xrenya/pvt-small-224")
        img_hw = 224
        model_name = "pvt_small"
    elif model_select == 18:
        model = PvtForImageClassification.from_pretrained("Zetatech/pvt-medium-224")
        img_hw = 224
        model_name = "pvt_med"

    model = model.eval()
    model.to("cuda:0")

    # transform data into format needed for resnet model model expects 224x224 3-color image
    transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])
    
    ''' These three files map the imagenet validation images to their class names and their class index
        gnd_truth holds the line number in class_map that the corresponding image's class name is on
        class_list holds the class names at the line number that represents the images index 
    '''
    gnd_truth = np.loadtxt("../class_maps/ImageNet/ILSVRC2012_validation_ground_truth.txt").astype(np.int64)

    clsloc = open("../class_maps/ImageNet/map_clsloc.txt", 'r')
    class_map = clsloc.readlines()

    classes = open("../class_maps/ImageNet/imagenet_classes.txt", 'r')
    class_list = classes.readlines()
    
    images_seen = 0
    
    correctly_classified = np.zeros(50000, dtype = np.int64)

    img_dir = "../../../ImageNet/"
    image_files = sorted(os.listdir(img_dir))

    batch_size = 100

    # --- Processing Loop ---
    for i in tqdm(range(0, len(image_files), batch_size), desc = model_name):
        batch_files = image_files[i : i + batch_size]
        batch_imgs = []
        batch_indices = []

        # --- Gather valid images---
        for j, image_name in enumerate(batch_files):
            try:
                img = Image.open(os.path.join(img_dir, image_name)).convert("RGB")
            except:
                continue

            img_tensor = transform(img)

            if img_tensor.shape != (3, img_hw, img_hw):
                images_seen += 1
                continue

            # Normalize and batch
            input_tensor = transform_normalize(img_tensor)
            
            batch_imgs.append(input_tensor)
            batch_indices.append(images_seen)
            images_seen += 1

        if len(batch_imgs) == 0:
            continue

        batch_tensor = torch.stack(batch_imgs).to("cuda:0")  # shape: (B, 3, H, W)

        # --- Classify ---
        with torch.no_grad():
            target_classes = getClass(batch_tensor, model, "cuda:0").cpu().tolist()

        # --- Compare with Ground Truth ---
        for idx, pred_class in zip(batch_indices, target_classes):
            # the line number + 1 of the class as listed in class_map
            map_line = gnd_truth[idx]
            # an id, line number, and class name e.g sea_snake
            class_info = class_map[map_line - 1]
            # get the class name, and replace the underscore(s) with space(s)
            class_name = class_info.split(" ")[-1].replace('_', ' ')
            # get the array number of the class name which is the class index
            true_class_index = class_list.index(class_name)

            if pred_class == true_class_index:
                correctly_classified[idx] = 1
    
    # write the list of 0s and 1s where a 1 represents a correctly classified image
    file = open("../class_maps/ImageNet/correctly_classified_" + model_name + ".txt", "w")
    size = len(correctly_classified)

    for i in range(size):
        file.write(str(correctly_classified[i]) + "\n")
    file.close()
    
    return

if __name__ == "__main__":
    main()