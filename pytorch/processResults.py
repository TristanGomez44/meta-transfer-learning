import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torchvision
import sys
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def normMap(map,onlyMax=False):
    map_min = map.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0].min(dim=-3,keepdim=True)[0]
    map_max = map.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0].max(dim=-3,keepdim=True)[0]

    if not onlyMax:
        map = (map-map_min)/(map_max-map_min)
    else:
        map = map/map_max
    return map

def reshape(tens,smooth=True):
    if smooth:
        tens = torch.nn.functional.interpolate(tens,(300,300),mode="bilinear",align_corners=False)
    else:
        tens = torch.nn.functional.interpolate(tens,(300,300),mode="nearest")
    tens = tens.unsqueeze(-1)
    tens = tens.reshape(-1,5,tens.size(1),tens.size(2),tens.size(3))
    tens = tens.permute(1,0,2,3,4)
    return tens

def applyCMap(tens,cmPlasma):
    tens = torch.tensor(cmPlasma(tens[0,0].numpy())[:,:,:3]).float()
    tens = tens.permute(2,0,1).unsqueeze(0)
    return tens

def loadNorm(model_id,ind,suff,smooth=False):
    norms = torch.load("../results/cifar/{}_normQuer{}{}.th".format(model_id,ind*5,suff),map_location="cpu")
    norms = normMap(norms,onlyMax=False)
    norms = reshape(norms,smooth=smooth)
    return norms

def loadAttNorm(model_id,ind,suff,smooth=False):
    attMaps = torch.load("../results/cifar/{}_simMapQuer{}{}.th".format(model_id,ind*5,suff),map_location="cpu")
    attMaps = torch.cat(attMaps[:3],dim=1)
    attMaps = normMap(attMaps)

    norms = torch.load("../results/cifar/{}_normQuer{}{}.th".format(model_id,ind*5,suff),map_location="cpu")
    norms = normMap(norms,onlyMax=False)
    attMaps = attMaps*norms
    attMaps = reshape(attMaps,smooth=smooth)
    norms = reshape(norms,smooth=smooth)
    return attMaps,norms

def mixAndCat(catImg,map,img):
    mix = 0.8*map+0.2*img.mean(dim=1,keepdim=True)
    return torch.cat((catImg,mix),dim=0)

def visMaps(gradcam_id,model_id,high_res_id,nrows,classes_to_plot,img_to_plot,plot_id,batch_inds_to_plot,test_on_val,nb_per_class):

    inv_normalize = torchvision.transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.255])

    suff = "_val" if test_on_val else ""

    batchPaths = glob.glob("../results/cifar/{}_dataQuer*{}.th".format(high_res_id,suff))

    if not test_on_val:
        batchPaths = list(filter(lambda x:x.find("_val") == -1,batchPaths))

    batchInds = list(map(lambda x:int(x.split("Quer")[1].split("{}.th".format(suff))[0]),batchPaths))
    batchInds = sorted(batchInds)

    cmPlasma = plt.get_cmap('plasma')

    if not img_to_plot is None:
        catImg = None

    for i,ind in enumerate(batch_inds_to_plot):

        gradcams = torch.load("../results/cifar/{}_gradcamQuer{}{}.th".format(gradcam_id,ind*5,suff),map_location="cpu")
        gradcams = reshape(normMap(gradcams))

        gradcams_pp = torch.load("../results/cifar/{}_gradcamppQuer{}{}.th".format(gradcam_id,ind*5,suff),map_location="cpu")
        gradcams_pp = reshape(normMap(gradcams_pp))

        guideds = torch.load("../results/cifar/{}_guidedQuer{}{}.th".format(gradcam_id,ind*5,suff),map_location="cpu")
        guideds = normMap(guideds)
        guideds = torch.abs(guideds - guideds.mean())
        guideds = guideds.mean(dim=1,keepdims=True)
        #guideds = torch.nn.functional.max_pool2d(guideds,2)

        guideds= reshape(normMap(guideds))
        guideds *= gradcams_pp
        guideds = normMap(guideds)

        attMaps,_ = loadAttNorm(model_id,ind,suff,smooth=True)
        norms = loadNorm(gradcam_id,ind,suff,smooth=True)
        attMaps_hr,norms_hr = loadAttNorm(high_res_id,ind,suff)

        imgs = torch.load("../results/cifar/{}_dataQuer{}{}.th".format(model_id,ind*5,suff),map_location="cpu")

        newImgs = []
        for img in imgs:
            img = inv_normalize(img)
            newImgs.append(img.unsqueeze(0))

        imgs = torch.cat(newImgs,dim=0)
        imgs = torch.tensor(imgs.numpy().copy())

        imgs = reshape(imgs)

        if img_to_plot is None:
            catImg = None

        for j in range(len(imgs)):

            if classes_to_plot is None or j == classes_to_plot[i]:

                nbPlot = len(imgs[j])

                for k in range(nbPlot):

                    if img_to_plot is None or k == img_to_plot[i]:

                        if catImg is None:
                            catImg = imgs[j][k].unsqueeze(0)
                        else:
                            catImg = torch.cat((catImg,imgs[j][k].unsqueeze(0)),dim=0)

                        print(gradcams[j][k].min(),gradcams[j][k].mean(),gradcams[j][k].max())

                        gradcam = applyCMap(gradcams[j][k].unsqueeze(0).detach(),cmPlasma)
                        gradcam_pp = applyCMap(gradcams_pp[j][k].unsqueeze(0).detach(),cmPlasma)
                        guided = applyCMap((guideds[j][k]).unsqueeze(0).detach(),cmPlasma)
                        norm = applyCMap(norms[j][k].unsqueeze(0).detach(),cmPlasma)
                        catImg = mixAndCat(catImg,guided,imgs[j][k].unsqueeze(0))
                        catImg = mixAndCat(catImg,gradcam,imgs[j][k].unsqueeze(0))
                        catImg = mixAndCat(catImg,gradcam_pp,imgs[j][k].unsqueeze(0))
                        catImg = mixAndCat(catImg,norm,imgs[j][k].unsqueeze(0))

                        attMaps[j][k][0] = (attMaps[j][k][0]-attMaps[j][k][0].min())/(attMaps[j][k][0].max()-attMaps[j][k][0].min())
                        catImg = mixAndCat(catImg,attMaps[j][k].unsqueeze(0),imgs[j][k].unsqueeze(0))

                        attMaps_hr[j][k] = (attMaps_hr[j][k]-attMaps_hr[j][k].min())/(attMaps_hr[j][k].max()-attMaps_hr[j][k].min())
                        catImg = mixAndCat(catImg,attMaps_hr[j][k].unsqueeze(0),imgs[j][k].unsqueeze(0))

        if img_to_plot is None:
            outPath = "../vis/cifar/{}_attMaps_batch{}_{}{}.png".format(model_id,i,plot_id,suff)
            torchvision.utils.save_image(catImg,outPath,nrow=nrows)
            #os.system("convert -resize 20% {} {}".format(outPath,outPath.replace(".png","_small.png")))

    if not img_to_plot is None:
        outPath = "../vis/cifar/{}_attMaps_{}{}.png".format(model_id,plot_id,suff)
        torchvision.utils.save_image(catImg,outPath,nrow=nrows)
        #os.system("convert -resize 20% {} {}".format(outPath,outPath.replace(".png","_small.png")))

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gradcam_id',type=str)
    parser.add_argument('--model_id',type=str)
    parser.add_argument('--high_res_id',type=str)
    parser.add_argument('--nrows',type=int,default=6)
    parser.add_argument('--plot_id',type=str)
    parser.add_argument('--batch_inds_to_plot',type=int,nargs="*")
    parser.add_argument('--classes_to_plot',type=int,nargs="*")
    parser.add_argument('--img_to_plot',type=int,nargs="*")
    parser.add_argument('--test_on_val',action='store_true')
    parser.add_argument('--nb_per_class',type=int)
    args = parser.parse_args()

    visMaps(args.gradcam_id,args.model_id,args.high_res_id,args.nrows,args.classes_to_plot,args.img_to_plot,args.plot_id,args.batch_inds_to_plot,args.test_on_val,args.nb_per_class)
