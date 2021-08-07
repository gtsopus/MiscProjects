#Simple utility script using cv2 to convert .ppm images to .png

import cv2 as cv
import sys

def splitFileNames(args):
    newArgs = []

    for i in args:
        filename = i.split(".")
        if(filename[1] != "ppm"):
            print("Error: only .ppm Files must be given as inputs!")
            sys.exit()
        else:
            newArgs.append(filename[0])

    return newArgs

def cv2PPM2PNG(args):
    filenames = splitFileNames(args)

    for i in filenames:
        image = cv.imread(i+'.ppm')
        cv.imwrite(i+'.png',image)
    
    print(str(len(filenames)) + " .ppm files successfully converted!")

def main(args):
    args = args[1:] #skip filename
    cv2PPM2PNG(args)

if __name__ == '__main__':
      main(sys.argv)