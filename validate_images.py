import cv2
import os
from pathlib import Path

def validate_images(directory, folders):

    #for i in ["train", "test", "valid"]:

    for f in folders:

        direc = directory + "\\" + "\\" + f + "\\"

        print(direc)

        for filename in os.listdir(direc):

            imagePath = direc + filename
            delete = False
         
            # try to load the image
            try:
                image = cv2.imread(imagePath)
         
                # if the image is `None` then we could not properly load it
                # from disk, so delete it
                if image is None:
                    delete = True
         
            # if OpenCV cannot load the image then the image is likely
            # corrupt so we should delete it
            except:
                print("Except")
                delete = True
         
            # check to see if the image should be deleted
            if delete:
                print("[INFO] deleting {}".format(imagePath))
                os.remove(imagePath)

if __name__ == '__main__':

    directory = os.getcwd() + "\\data\\images\\christmas" 
    validate_images(directory, folders = ["santa", "tree", "raindeer"])