import os
import random

def assign_pics_to_folders(folders_root, directory_to, folders):
    
    for directory in folders:

        directory_from = folders_root + directory
        
        for file in os.listdir(directory_from):

            val = random.uniform(0, 1)
            
            if val > 0.9:

                direc = directory_to + "test" + "/" + directory + "/" + file
                os.rename(directory_from + "/" + file, direc)

            elif val > 0.7:

                direc = directory_to + "valid" + "/" + directory + "/" + file
                os.rename(directory_from + "/" + file, direc)

            else:

                direc = directory_to + "train" + "/" + directory + "/" + file
                os.rename(directory_from + "/" + file, direc)





if __name__ == '__main__':

    assign_pics_to_folders("C:/Users/janne.m.flinck/Desktop/christmas/data/images/christmas/",
        "C:/Users/janne.m.flinck/Desktop/christmas/data/images/christmas/",
        folders = ["santa", "tree", "reindeer"])