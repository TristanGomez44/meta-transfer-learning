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
""" Trainer for meta-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset
from gradcam import GradCAM
from gradcam import GradCAMpp
from guided_backprop import GuidedBackprop
class MetaTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type, 'MTL'])
        save_path2 = 'shot' + str(args.shot) + '_way' + str(args.way) + '_query' + str(args.train_query) + \
            '_step' + str(args.step_size) + '_gamma' + str(args.gamma) + '_lr1' + str(args.meta_lr1) + '_lr2' + str(args.meta_lr2) + \
            '_batch' + str(args.num_batch) + '_maxepoch' + str(args.max_epoch) + \
            '_baselr' + str(args.base_lr) + '_updatestep' + str(args.update_step) + \
            '_stepsize' + str(args.step_size) + '_' + args.meta_label
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load meta-train set
        self.trainset = Dataset('train', self.args)
        self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.way, self.args.shot + self.args.train_query)
        self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=args.num_workers, pin_memory=True)

        # Load meta-val set
        self.valset = Dataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.label, 600, self.args.way, self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=args.num_workers, pin_memory=True)

        # Build meta-transfer learning model
        self.model = MtlLearner(self.args,res="high" if (self.args.distill_id or self.args.high_res) else "low",multi_gpu=len(args.gpu.split(","))>1)

        if self.args.distill_id:
            #self.teacher = MtlLearner(self.args,res="low")
            #self.teacher.load_state_dict(torch.load(args.distill_id)["params"])

            self.teacher = MtlLearner(self.args,res="low",repVecNb=self.args.nb_parts_teach,multi_gpu=len(args.gpu.split(","))>1)
            bestTeach = "../models/{}/meta_{}_trial{}_max_acc.pth".format(self.args.exp_id,self.args.distill_id,self.args.best_trial_teach-1)
            self.teacher.load_state_dict(torch.load(bestTeach)["params"])

        # Set optimizer
        self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
            {'params': self.model.base_learner.parameters(), 'lr': self.args.meta_lr2}], lr=self.args.meta_lr1)
        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        # load pretrained model without FC classifier
        self.model_dict = self.model.state_dict()
        if self.args.init_weights is not None:
            pretrained_dict = torch.load(self.args.init_weights)['params']

            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}

            self.model_dict.update(pretrained_dict)
            self.model.load_state_dict(self.model_dict)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

            if self.args.distill_id:
                self.teacher = self.teacher.cuda()

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        #torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))
        torch.save(dict(params=self.model.state_dict()), "../models/{}/meta_{}_trial{}_{}.pth".format(self.args.exp_id,self.args.model_id,self.args.trial_number,name))

    def train(self,trial):
        """The function for the meta-train phase."""

        # Set the meta-train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Generate the labels for train set of the episodes
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)

        worstClasses = []

        # Start meta-train
        for epoch in range(1, self.args.max_epoch + 1):
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # Generate the labels for test set of the episodes during meta-train updates
            label = torch.arange(self.args.way).repeat(self.args.train_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data,targ = [_.cuda() for _ in batch]
                else:
                    data,targ = batch
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                # Output logits for model
                logits = self.model((data_shot, label_shot, data_query))
                # Calculate meta-train loss
                loss = F.cross_entropy(logits, label)
                # Calculate meta-train accuracy

                if self.args.distill_id:
                    teachLogits = self.teacher((data_shot, label_shot, data_query))
                    kl = F.kl_div(F.log_softmax(logits/self.args.kl_temp, dim=1),F.softmax(teachLogits/self.args.kl_temp, dim=1),reduction="batchmean")
                    loss = (kl*self.args.kl_interp*self.args.kl_temp*self.args.kl_temp+loss*(1-self.args.kl_interp))

                acc = count_acc(logits, label)
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.args.hard_tasks:
                    if len(worstClasses) == self.args.way:
                        inds = self.train_sampler.hardBatch(worstClasses)
                        batch = [self.trainset[i][0] for i in inds]
                        data_shot, data_query = data[:p], data[p:]
                        logits = self.model((data_shot, label_shot, data_query))
                        loss = F.cross_entropy(logits, label)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        worstClasses = []
                    else:
                        error_mat = (logits.argmax(dim=1) == label).view(self.args.train_query,self.args.way)
                        worst = error_mat.float().mean(dim=0).argmin()
                        worst_trueInd = targ[worst]
                        worstClasses.append(worst_trueInd)

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # Start validation for this epoch, set model to eval mode
            self.model.eval()

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # Generate the labels for test set of the episodes during meta-val for this epoch
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            # Print previous information
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            # Run meta-validation
            for i, batch in enumerate(self.val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                logits = self.model((data_shot, label_shot, data_query))
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)

                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)
            # Print loss and accuracy for this epoch
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc_averager))

            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))

            trial.report(val_acc_averager,epoch)

        writer.close()

    def eval(self,gradcam=False,test_on_val=False):
        """The function for the meta-eval phase."""
        # Load the logs
        if os.path.exists(osp.join(self.args.save_path, 'trlog')):
            trlog = torch.load(osp.join(self.args.save_path, 'trlog'))
        else:
            trlog = None

        torch.manual_seed(1)
        np.random.seed(1)
        # Load meta-test set
        test_set = Dataset('val' if test_on_val else 'test', self.args)
        sampler = CategoriesSampler(test_set.label, 600, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

        # Set test accuracy recorder
        test_acc_record = np.zeros((600,))

        # Load model for meta-test phase
        if self.args.eval_weights is not None:
            weights = self.addOrRemoveModule(self.model,torch.load(self.args.eval_weights)['params'])
            self.model.load_state_dict(weights)
        else:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc' + '.pth'))['params'])
        # Set model to eval mode
        self.model.eval()

        # Set accuracy averager
        ave_acc = Averager()

        # Generate labels
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)

        if gradcam:
            self.model.layer3 = self.model.encoder.layer3
            model_dict = dict(type="resnet", arch=self.model, layer_name='layer3')
            grad_cam = GradCAM(model_dict, True)
            grad_cam_pp = GradCAMpp(model_dict, True)
            self.model.features = self.model.encoder
            guided = GuidedBackprop(self.model)

        # Start meta-test
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = self.args.way * self.args.shot
            data_shot, data_query = data[:k], data[k:]

            if i % 5 == 0:
                suff = "_val" if test_on_val else ""
                if self.args.rep_vec:
                    print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
                    logits,simMapQuer,simMapShot,normQuer,normShot,fast_weights = self.model((data_shot, label_shot, data_query),retSimMap=True)

                    torch.save(simMapQuer,"../results/{}/{}_simMapQuer{}{}.th".format(self.args.exp_id,self.args.model_id,i,suff))
                    torch.save(simMapShot,"../results/{}/{}_simMapShot{}{}.th".format(self.args.exp_id,self.args.model_id,i,suff))
                    torch.save(data_query,"../results/{}/{}_dataQuer{}{}.th".format(self.args.exp_id,self.args.model_id,i,suff))
                    torch.save(data_shot,"../results/{}/{}_dataShot{}{}.th".format(self.args.exp_id,self.args.model_id,i,suff))
                    torch.save(normQuer,"../results/{}/{}_normQuer{}{}.th".format(self.args.exp_id,self.args.model_id,i,suff))
                    torch.save(normShot,"../results/{}/{}_normShot{}{}.th".format(self.args.exp_id,self.args.model_id,i,suff))
                else:
                    logits,normQuer,normShot,fast_weights = self.model((data_shot, label_shot, data_query),retFastW=True,retNorm=True)
                    torch.save(normQuer,"../results/{}/{}_normQuer{}{}.th".format(self.args.exp_id,self.args.model_id,i,suff))
                    torch.save(normShot,"../results/{}/{}_normShot{}{}.th".format(self.args.exp_id,self.args.model_id,i,suff))

                print("Saving gradmaps",i)
                allMasks,allMasks_pp,allMaps = [],[],[]
                for l in range(len(data_query)):
                    allMasks.append(grad_cam(data_query[l:l+1],fast_weights,None))
                    allMasks_pp.append(grad_cam_pp(data_query[l:l+1],fast_weights,None))
                    allMaps.append(guided.generate_gradients(data_query[l:l+1],fast_weights))
                allMasks = torch.cat(allMasks,dim=0)
                allMasks_pp = torch.cat(allMasks_pp,dim=0)
                allMaps = torch.cat(allMaps,dim=0)

                torch.save(allMasks,"../results/{}/{}_gradcamQuer{}{}.th".format(self.args.exp_id,self.args.model_id,i,suff))
                torch.save(allMasks_pp,"../results/{}/{}_gradcamppQuer{}{}.th".format(self.args.exp_id,self.args.model_id,i,suff))
                torch.save(allMaps,"../results/{}/{}_guidedQuer{}{}.th".format(self.args.exp_id,self.args.model_id,i,suff))

            else:
                logits = self.model((data_shot, label_shot, data_query))

            acc = count_acc(logits, label)
            ave_acc.add(acc)
            test_acc_record[i-1] = acc

        # Calculate the confidence interval, update the logs
        m, pm = compute_confidence_interval(test_acc_record)
        if trlog is not None:
            print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))

        return m

    def addOrRemoveModule(self,net,weights):

        exKeyWei = None
        for key in weights:
            if key.find("encoder") != -1:
                exKeyWei = key
                break
            else:
                print(key)

        exKeyNet = None
        for key in net.state_dict():
            if key.find("encoder") != -1:
                exKeyNet = key
                break

        print(exKeyWei,exKeyNet)

        if exKeyWei.find("module") != -1 and exKeyNet.find("module") == -1:
            #remove module
            newWeights = {}
            for param in weights:
                newWeights[param.replace("module.","")] = weights[param]
            weights = newWeights

        if exKeyWei.find("module") == -1 and exKeyNet.find("module") != -1:
            #add module
            newWeights = {}
            for param in weights:
                if param.find("encoder") != -1:
                    param_split = param.split(".")
                    newParam = param_split[0]+"."+"module."+".".join(param_split[1:])
                    newWeights[newParam] = weights[param]
                else:
                    newWeights[param] = weights[param]
            weights = newWeights

        return weights
