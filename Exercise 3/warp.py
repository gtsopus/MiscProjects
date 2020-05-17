#TSOPOURIDIS GRIGORIOS, AM : 3358
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

imagePoints = list()
pointsCount = 0


#get 4 points from an image on mouse click up and then close cv2 window
def getImagePoint(event, x,y,flags,param):
    global pointsCount, imagePoints
    if(event == cv2.EVENT_LBUTTONUP and pointsCount<4):
        imagePoints.append((x,y))
        pointsCount += 1
    if(pointsCount >= 4):
        cv2.destroyAllWindows()

#warp an image using 4 points taken from users input (mouse click)
def warp(inputFileName, outputFileName):
    global imagePoints
    print("#P1------P3\n|        |\n#P2------P4")
    img = cv2.imread(inputFileName)
    #get image points
    cv2.namedWindow('Image',cv2.WINDOW_FREERATIO)
    cv2.imshow('Image',img)
    cv2.setMouseCallback("Image", getImagePoint)
    cv2.waitKey(0)
    height, width, channels = img.shape
    #calculate perspective transformation matrix
    destPoints = [(0,0),(0,height-1),(width-1,0),(width-1,height-1)]
    

    print("Selected points: ")
    print(imagePoints)
    print("Destination points: ")
    print(destPoints)
    pts1 = np.array(imagePoints)
    pts2 = np.array(destPoints)

    #T((xi,yi)) = (xi',yi')
    #[a1 a2 a3][xi]   [a1xi + a2yi + a3]
    #[a4 a5 a6][yi] = [a4xi + a5yi + a6]
    #[a7 a8 1 ][ 1]   [a7xi + a8yi + 1 ]
    #       a1xi + a2yi + a3
    #xi' = ------------------
    #       a7xi + a8yi + 1
    #
    #       a4xi + a5yi + a6
    #yi' = ------------------
    #       a7xi + a8yi + 1
    #linear equations = we get this matrix form:

    A = [[pts1[0][0],pts1[0][1],1,0,0,0,-pts1[0][0]*pts2[0][0],-pts1[0][1]*pts2[0][0]],
        [0,0,0,pts1[0][0],pts1[0][1],1,-pts1[0][0]*pts2[0][1],-pts1[0][1]*pts2[0][1]],
        [pts1[1][0],pts1[1][1],1,0,0,0,-pts1[1][0]*pts2[1][0],-pts1[1][1]*pts2[1][0]],
        [0,0,0,pts1[1][0],pts1[1][1],1,-pts1[1][0]*pts2[1][1],-pts1[1][1]*pts2[1][1]],
        [pts1[2][0],pts1[2][1],1,0,0,0,-pts1[2][0]*pts2[2][0],-pts1[2][1]*pts2[2][0]],
        [0,0,0,pts1[2][0],pts1[2][1],1,-pts1[2][0]*pts2[2][1],-pts1[2][1]*pts2[2][1]],
        [pts1[3][0],pts1[3][1],1,0,0,0,-pts1[3][0]*pts2[3][0],-pts1[3][1]*pts2[3][0]],
        [0,0,0,pts1[3][0],pts1[3][1],1,-pts1[3][0]*pts2[3][1],-pts1[3][1]*pts2[3][1]]]
    

    A = np.asarray(A)
    A = np.float32(A)

    #A * x = b
    b = [[pts2[0][0]],[pts2[0][1]],[pts2[1][0]],[pts2[1][1]],[pts2[2][0]],[pts2[2][1]],[pts2[3][0]],[pts2[0][1]]]
    b = np.asarray(b)
    
    #x = perspective matrix
    #np.linalg.inv, inverse matrix

    #n=4 and determinant != 0
    if(np.linalg.det(A)!=0):
        x = np.matmul(np.linalg.inv(A),b)
    #correct matrix format
    perspM = [[x[0],x[1],x[2]],
              [x[3],x[4],x[5]],
              [x[6],x[7],1]]
    perspM = np.array(perspM)
    perspM = np.float32(perspM)
    
    print("Perspective transform matrix: ")
    print(perspM)

    img = cv2.warpPerspective(img,perspM,(width,height))

    img = cv2.resize(img,(1000,1000))
    #plt in order for image to show in notebook
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite(outputFileName,img)

if __name__ == "__main__":
    if __name__ == "__main__":
        if(len(sys.argv) != 3):
            print("Incorrect format.")
            print("Please type : python3 warp.py <input filename> <output filename>")
        else:
            warp(sys.argv[1], sys.argv[2])
    