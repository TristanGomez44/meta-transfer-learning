##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/Sha-Lab/FEAT
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" ResNet with MTL. """
import torch.nn as nn
import torch
from models.conv2d_mtl import Conv2dMtl

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv1x1mtl(in_planes, out_planes, stride=1):
    return Conv2dMtl(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False)

def conv3x3mtl(in_planes, out_planes, stride=1):
    return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlockMtl(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockMtl, self).__init__()
        self.conv1 = conv3x3mtl(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3mtl(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckMtl(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckMtl, self).__init__()
        self.conv1 = Conv2dMtl(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dMtl(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2dMtl(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetMtl(nn.Module):

    def __init__(self, layers=[4, 4, 4], mtl=True,repVec=True,nbVec=3,res="high",repvec_merge=False,b_cnn=False):
        super(ResNetMtl, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl
            block = BasicBlockMtl
            self.conv1x1 = conv1x1mtl
        else:
            self.Conv2d = nn.Conv2d
            block = BasicBlock
            self.conv1x1 = conv1x1

        cfg = [160, 320, 640]
        self.inplanes = iChannels = int(cfg[0]/2)
        self.conv1 = self.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, cfg[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, cfg[1], layers[1], stride=2 if res=="low" else 1)
        self.layer3 = self._make_layer(block, cfg[2], layers[2], stride=2 if res=="low" else 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.nbVec = nbVec
        self.repVec = repVec
        self.res = res
        for m in self.modules():
            if isinstance(m, self.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.repvec_merge = repvec_merge
        if repvec_merge:

            if self.nbVec == 3:
                half = nn.Linear(cfg[2],cfg[2]//2)
                quarter = nn.Linear(cfg[2],cfg[2]//4)
                self.vec1 = half
                self.vec2 = quarter
                self.vec3 = quarter
            elif self.nbVec == 5:
                quarter = nn.Linear(cfg[2],cfg[2]//4)
                eigth = nn.Linear(cfg[2],cfg[2]//8)
                self.vec1 = quarter
                self.vec2 = quarter
                self.vec3 = quarter
                self.vec4 = eigth
                self.vec5 = eigth
            elif self.nbVec == 7:
                quarter = nn.Linear(cfg[2],cfg[2]//4)
                eigth = nn.Linear(cfg[2],cfg[2]//8)
                self.vec1 = quarter
                self.vec2 = eigth
                self.vec3 = eigth
                self.vec4 = eigth
                self.vec5 = eigth
                self.vec6 = eigth
                self.vec7 = eigth
            else:
                raise ValueError("Wrong part number for merge : ",self.nbVec)


        self.b_cnn = b_cnn
        if self.b_cnn:
            attention = [BasicBlock(cfg[-1], cfg[-1])]
            attention.append(conv1x1(cfg[-1], nbVec))
            attention.append(nn.ReLU())
            self.att = nn.Sequential(*attention)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def compAtt(self,x):
        attMaps = self.att(x)
        x = (attMaps.unsqueeze(2)*x.unsqueeze(1)).reshape(x.size(0),x.size(1)*(attMaps.size(1)),x.size(2),x.size(3))
        x = self.avgpool(x)
        return x.view(x.size(0), -1),[attMaps[:,i:i+1] for i in range(attMaps.size(1))]

    def forward(self, x,retSimMap=False,retNorm=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.res == "high":
            x = x[:,:,2:-2,2:-2]

        if retSimMap or retNorm:
            norm = torch.sqrt(torch.pow(x,2).sum(dim=1,keepdim=True))

        if self.repVec:
            if not self.b_cnn:
                x,simMap = representativeVectors(x,self.nbVec)
                if self.repvec_merge:
                    finalVec = []
                    for k,vec in enumerate(x):
                        lin = getattr(self,"vec{}".format(k+1))
                        outVec = lin(vec)
                        finalVec.append(outVec)
                    x = torch.cat(finalVec,dim=-1)
                else:
                    x = torch.cat(x,dim=-1)

            else:
                x,simMap= self.compAtt(x)

        else:
            x = self.avgpool(x)
            x = x.view(x.size(0),-1)

        if retSimMap:
            retDict = {"x":x,"simMap":simMap,"norm":norm}
            return retDict
        elif retNorm:
            retDict = {"x":x,"norm":norm}
            return retDict
        else:
            return x

def representativeVectors(x,nbVec):

    xOrigShape = x.size()

    x = x.permute(0,2,3,1).reshape(x.size(0),x.size(2)*x.size(3),x.size(1))
    norm = torch.sqrt(torch.pow(x,2).sum(dim=-1)) + 0.00001

    raw_reprVec_score = norm.clone()

    repreVecList = []
    simList = []
    for _ in range(nbVec):
        _,ind = raw_reprVec_score.max(dim=1,keepdim=True)
        raw_reprVec_norm = norm[torch.arange(x.size(0)).unsqueeze(1),ind]
        raw_reprVec = x[torch.arange(x.size(0)).unsqueeze(1),ind]
        sim = (x*raw_reprVec).sum(dim=-1)/(norm*raw_reprVec_norm)
        simNorm = sim/sim.sum(dim=1,keepdim=True)
        reprVec = (x*simNorm.unsqueeze(-1)).sum(dim=1)
        repreVecList.append(reprVec)
        raw_reprVec_score = (1-sim)*raw_reprVec_score
        simReshaped = simNorm.reshape(sim.size(0),1,xOrigShape[2],xOrigShape[3])
        simList.append(simReshaped)

    return repreVecList,simList
