<h1>EVALUATION ON ALL MODELS</h1>

`cd XAI_Survey/evaluations`

You should have the imagenet ILSVRC2012 validation set of images in a folder to point these tests to.

You should also have the [ImageNet segmentation dataset](http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat) as a .mat file (click link to download).

The code by default expects these two sets of images to be in the main directory. e.g. ../../dataset

All quanitative tests are contained in the allPertTests.txt, allSanityTests.txt, and allSegTests.txt files. Run these in the command line to run all tests.
