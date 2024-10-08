import cv2
import os
import time

frameWidth = 640
frameHeight = 480

path = "Data Collection/"

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
numImages = 250

for label in labels:
    n = 1
    os.mkdir(path + label)
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    print('Collecting images for {}' .format(label))
    time.sleep(2)

    while n < numImages:
        _, image = cap.read(0)
        img = cv2.flip(image, 1)
        greyImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgName = os.path.join(path + label, label+'_'+'{}.jpg' .format(str(n)))
        blur = cv2.Laplacian(image, cv2.CV_64F).var()

        if blur > 50:
            n += 1
            cv2.imwrite(imgName, greyImage)
            cv2.imshow("Result", greyImage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()