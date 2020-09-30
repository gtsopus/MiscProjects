## The following files were created for university assignments during the academic year of 2019-2020 for my "Computer Vision" course. The results may be incorrect or unexpected.

### Files : 

1. Ask1.py - rename to threshold.py. Simple image thresholding given a specific threshold by the user. 
2. Ask2.py - rename to affinetrans.py. Applies an affine transformation on a given image by taking in 6 parameters by the user. (affine matrix first 6 elements).

3. adaptive.py - Use of otsu thresholding algorithm for adaptive thresholding by calculating pixel-specific otsu-threshold from a sub-image (window). Window size of given by the user.

4. warp.py - Apply perspective transformation to an image, by calculating the transformation matrix manually (task of the assignment), given 4 points clicked in by the user.

5. cv_semantic.py - Uses the deeplab mobilenet with PASCAL VOC dataset in order to find the semantic segmantation of an image, get a specific layer output PCA it to 3 channel image (RGB) and show it and get a specific layer output, PCA it to 8 N x M x 8 data array and use kmeans to binarize it.

### Dependencies:
(Number corresponds to the same file-number)
1. numpy, matplotlib, Pillow
2. numpy, matplotlib, Pillow
3. numpy, matplotlib, Pillow
4. numpy, matplotlib, Pillow, cv2
5. numpy, sklearn, tensorflow, deeplab, matplotlib, six.moves, Pillow

Made by: Grigorios Tsopouridis,
All credits of imported libraries/models go to their respective owners.
Deeplab code modeled after the deeplab_demo notebook.
