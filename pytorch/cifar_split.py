import glob
import numpy as np
import os
import shutil


root = "datasets/cifar-FS/"

splitFolder = os.path.join(root,"splits/bertinetto/")

sets = ["train","val","test"]

for set in sets:
    trainClasses = np.genfromtxt(os.path.join(splitFolder,set+".txt"),dtype=str)

    if not os.path.exists(os.path.join(root,set)):
        os.makedirs(os.path.join(root,set))

    for className in trainClasses:
        shutil.move(os.path.join(root,"data",className),os.path.join(root,set))
        #print(os.path.join(root,"data",className),"go to",os.path.join(root,set))
