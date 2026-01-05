from torchvision import models
from torchvision import transforms
from PIL import Image
import os, sys
import numpy as np
import torch

os.sys.path.append(os.path.dirname(os.path.abspath('.')))

def main():
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
    
    file_items = []
    
    for image in sorted(os.listdir("../../../ImageNet")): 
        img = Image.open("../../../ImageNet/" + image)

        # the line number + 1 of the class as listed in class_map
        map_line = gnd_truth[images_seen]
        # an id, line number, and class name e.g sea_snake
        class_info = class_map[map_line - 1]
        # get the class name, and replace the underscore(s) with space(s)
        class_name = (class_info.split(" ")[-1]).replace('_', ' ')
        # get the array number of the class name which is the class index
        true_class_index = class_list.index(class_name)
        
        file_items.append(image + " " + str(true_class_index))
        
        images_seen += 1
    
    # write the list of 0s and 1s where a 1 represents a correctly classified image
    file = open("../class_maps/ImageNet/file_names_and_class.txt", "w")
    size = len(file_items)

    for i in range(size):
        file.write(file_items[i] + "\n")
    file.close()
    
    return

if __name__ == "__main__":
    main()