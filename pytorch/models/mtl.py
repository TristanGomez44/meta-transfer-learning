##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Model for meta-transfer learning. """
import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_mtl import ResNetMtl

class DataParallelModel(nn.DataParallel):
    def __init__(self, model):
        super(DataParallelModel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(DataParallelModel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            #the_vars = self.vars
            the_vars = [self.fc1_w,self.fc1_b]

        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        #return self.vars
        return [self.fc1_w,self.fc1_b]

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=64,res="high",repVecNb=None,multi_gpu=False):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        z_dim = 640
        self.repVec = args.rep_vec

        self.nbVec = args.nb_parts if repVecNb is None else repVecNb

        featNb = self.nbVec*z_dim if (args.rep_vec and not args.repvec_merge) else z_dim

        self.base_learner = BaseLearner(args,featNb)

        if self.mode == 'meta':
            self.encoder = ResNetMtl(repVec=args.rep_vec,nbVec=self.nbVec,res=res,repvec_merge=args.repvec_merge,b_cnn=args.b_cnn)
            self.pre_fc = None
        else:
            self.encoder = ResNetMtl(mtl=False,repVec=args.rep_vec,nbVec=self.nbVec,res=res,repvec_merge=args.repvec_merge,b_cnn=args.b_cnn)
            self.pre_fc = nn.Sequential(nn.Linear(featNb, 1000), nn.ReLU(), nn.Linear(1000, num_cls))

        if multi_gpu:
            self.encoder = DataParallelModel(self.encoder)
            #self.pre_fc = nn.DataParallel(self.pre_fc) if not self.pre_fc is None else None
            #self.base_learner = nn.DataParallel(self.base_learner)

    def forward(self, inp,retSimMap=False,retFastW=False,retNorm=False):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """
        if self.mode=='pre':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query,retSimMap,retFastW,retNorm)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        """The function to forward pretrain phase.
        Args:
          inp: input images.
        Returns:
          the outputs of pretrain model.
        """
        return self.pre_fc(self.encoder(inp))

    def meta_forward(self, data_shot, label_shot, data_query,retSimMap=False,retFastW=False,retNorm=False):
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        if retSimMap or retNorm:
            querDic = self.encoder(data_query,retSimMap,retNorm)
            shotDic = self.encoder(data_shot,retSimMap,retNorm)
            embedding_query = querDic["x"]
            embedding_shot = shotDic["x"]
        else:
            embedding_query = self.encoder(data_query,retSimMap)
            embedding_shot = self.encoder(data_shot,retSimMap)

        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)

        if retSimMap:
            return logits_q,querDic["simMap"],shotDic["simMap"],querDic["norm"],shotDic["norm"],fast_weights
        elif retFastW:
            if retNorm:
                return logits_q,querDic["norm"],shotDic["norm"],fast_weights
            else:
                return logits_q,fast_weights
            return logits_q,fast_weights
        else:
            return logits_q

    def preval_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, 100):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)
        return logits_q
