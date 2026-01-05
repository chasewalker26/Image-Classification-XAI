from torchvision import transforms
from PIL import Image
import os, sys
import numpy as np
import torch

import clip
import torch
from clip import tokenize
from torchvision.transforms import Resize
import torch.nn.functional as F


os.sys.path.append(os.path.dirname(os.path.abspath('.')))


def getClass(input, model, device):
    # calculate a prediction
    input = input.to(device)
    output = model(input)

    _, index = torch.max(output, 1)

    return index[0]

def main():

    img_hw = 224
    # clipmodel, preprocess = clip.load("ViT-B/16", device="cuda:0")
    clipmodel, preprocess = clip.load("ViT-B/32", device="cuda:0")
    clipmodel = clipmodel.eval()
    clipmodel.to("cuda:0")

    # crop and resize image
    transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.CenterCrop(img_hw),
    ])
    

    with open('../class_maps/ImageNet/ILSVRC2012_validation_ground_truth.txt') as f:
        gnd_truth = [int(line.strip()) for line in f.readlines()]
    with open('../class_maps/ImageNet/map_clsloc.txt') as f:
        class_map = [line.strip() for line in f.readlines()] 
    with open('../class_maps/ImageNet/imagenet_classes.txt') as f:
        class_list = [line.strip() for line in f.readlines()]

    clip_labels = [f"a photo of a {label}" for label in class_list]
    all_text_processed = clip.tokenize(clip_labels).cuda()
    all_classes_embedding = [clipmodel.encode_text(text.unsqueeze(0)).detach().cpu() for text in all_text_processed]
    all_classes_embedding = F.normalize(torch.from_numpy(np.array(all_classes_embedding)), dim=-1).to("cuda:0")

    images_seen = 0
    counter = 0
    
    correctly_classified = np.zeros(50000, dtype = np.int64)
    
    # for image in sorted(os.listdir("../../../../../ImageNet/ILSVRC2012_img_val/")): 
    #     img = Image.open("../../../../../ImageNet/ILSVRC2012_img_val/" + image)
    for image in sorted(os.listdir("../../../ImageNet")): 
        img = Image.open("../../../ImageNet/" + image)

        input_tensor = preprocess(transform(img))
        # only rgb images can be classified
        if input_tensor.squeeze().shape != (3, img_hw, img_hw):
            images_seen += 1
            continue

        # get imagenet ground truth class index and name
        map_line = gnd_truth[images_seen]
        class_info = class_map[map_line - 1]
        class_name = (class_info.split(" ")[-1]).replace('_', ' ')
        true_class_index = class_list.index(class_name)

        # get clip prediciton
        img_embedding = clipmodel.encode_image(input_tensor.unsqueeze(0).to("cuda:0"))
        similarities = img_embedding @ all_classes_embedding.squeeze().T
        pred = similarities.argmax().item()

        if pred == true_class_index:
            correctly_classified[images_seen] = 1
            counter += 1
        
        if images_seen % 100 == 0:
            print(str(images_seen) + " images done")
        
        images_seen += 1
    
    print(counter, "/", 50000, "images correct")

    # write the list of 0s and 1s where a 1 represents a correctly classified image
    file = open("../class_maps/ImageNet/correctly_classified_" + "CLIP_32" + ".txt", "w")
    size = len(correctly_classified)

    for i in range(size):
        file.write(str(correctly_classified[i]) + "\n")
    file.close()
    
    return

if __name__ == "__main__":
    main()