#TSOPOURIDIS GRIGORIOS AM: 3358
import sys
import numpy as np
#pillow (PIL) used to load the image from disk
from PIL import Image
import matplotlib.pyplot as plt

#Function that thresholds an image and saves it
def thresholdFunc(inputPath, outputPath, k):

    inImage = np.array(Image.open(inputPath)) # im2arr.shape: height x width x channel

    #check if image is RGB and convert it.
    if(inImage.ndim == 3):
        #3 channels, colored image
        #iterate through all image pixels and set them to the average of the RGB
        for i in range(len(inImage)):
            for j in range(len(inImage[0])):
                #use inImage[i][j][0] as the 1 channel (grayscale) image array converted from the rgb image
                inImage[i][j][0] = int((int(inImage[i][j][0]) + int(inImage[i][j][1]) + int(inImage[i][j][2]))/3)
        
        #keep only 1 dimension containing the luminocity values
        inImage = inImage[:,:,0]
   
    #inImage is grayscale now, handle both images the same way.

    #threshold the image
    #values > k are white
    #values =< k are black
    for i in range(len(inImage)):
        for j in range(len(inImage[0])):
            if(inImage[i][j] > k):
                #set pixel to white
                inImage[i][j] = 255
            else:
                #set pixel to black
                inImage[i][j] = 0

    #use PIL to convert the numpy array to image and then show it using matplot in order to put a label
    im = Image.fromarray(inImage)
    #plot the image
    plt.tight_layout()
    label = "Threshold: " + str(k)
    plt.xlabel(label)
    plt.imshow(inImage, cmap="gray")
    plt.show()
    #save the output image to the given file path
    im.save(outputPath)

#main function, parses console arguments
if __name__ == "__main__":
    if(len(sys.argv) != 4):
        print("Incorrect format.")
        print("Please type : python3 ask1.py <input filename> <output filename> <theshold k>")
    else:
        thresholdFunc(sys.argv[1], sys.argv[2], int(sys.argv[3]))
