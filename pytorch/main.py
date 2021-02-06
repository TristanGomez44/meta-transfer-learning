##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Main function for this repo. """
import argparse
import torch
from shutil import copyfile
import os
import configparser
import optuna
import sqlite3
import numpy as np
import gc

from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.meta import MetaTrainer
from trainer.pre import PreTrainer

def run(args,trial):

    #args.batch_size = trial.suggest_int("batch_size", 10, args.max_batch_size, log=True)
    #args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    #meta-train
    if args.phase == "meta_train":
        args.meta_lr1 = trial.suggest_float("meta_lr1",1e-5, 1e-3, log=True)
        args.meta_lr2 = trial.suggest_float("meta_lr2",1e-4, 1e-2, log=True)
        args.base_lr = trial.suggest_float("base_lr",1e-3, 1e-1, log=True)
        args.update_step = trial.suggest_int("update_step",10,100,log=True)
        args.step_size = trial.suggest_int("step_size",1,50,log=True)
        args.gamma = trial.suggest_float("gamma",0.1,0.9,step=0.2)

        #parser.add_argument('--meta_lr1', type=float, default=0.0001) # Learning rate for SS weights
        #parser.add_argument('--meta_lr2', type=float, default=0.001) # Learning rate for FC weights
        #parser.add_argument('--base_lr', type=float, default=0.01) # Learning rate for the inner loop
        #parser.add_argument('--update_step', type=int, default=50) # The number of updates for the inner loop
        #parser.add_argument('--step_size', type=int, default=10) # The number of epochs to reduce the meta learning rates
        #parser.add_argument('--gamma', type=float, default=0.5) # Gamma for the meta-train learning rate decay
    elif args.phase == "pre_train":
        # pretrain
        args.pre_batch_size = trial.suggest_int("pre_batch_size",8, args.max_batch_size, log=True)
        args.pre_lr = trial.suggest_float("pre_lr",1e-4, 1e-1, log=True)
        args.pre_gamma = trial.suggest_float("pre_gamma",0.05,0.25,step=0.05)
        args.pre_step_size = trial.suggest_int("pre_step_size",1,50, log=True)
        args.pre_custom_momentum = trial.suggest_float("pre_custom_momentum", 0.5, 0.99,log=True)
        args.pre_custom_weight_decay = trial.suggest_float("pre_custom_weight_decay", 1e-6, 1e-3, log=True)
        args.nb_parts = trial.suggest_int("nb_parts", 3, 64, log=True)

        #parser.add_argument('--pre_max_epoch', type=int, default=100) # Epoch number for pre-train phase
        #parser.add_argument('--pre_batch_size', type=int, default=128) # Batch size for pre-train phase
        #parser.add_argument('--pre_lr', type=float, default=0.1) # Learning rate for pre-train phase
        #parser.add_argument('--pre_gamma', type=float, default=0.2) # Gamma for the pre-train learning rate decay
        #parser.add_argument('--pre_step_size', type=int, default=30) # The number of epochs to reduce the pre-train learning rate
        #parser.add_argument('--pre_custom_momentum', type=float, default=0.9) # Momentum for the optimizer during pre-train
        #parser.add_argument('--pre_custom_weight_decay', type=float, default=0.0005) # Weight decay for the optimizer during pre-train
    else:
        raise ValueError("Unkown phase",args.phase)

    args.trial_number = trial.number

    if args.phase == "meta_train":
        bestPreTrialNb,args.nb_parts = findBestTrial(args,pre=True)
        args.init_weights = "../models/{}/pre_{}_trial{}_max_acc.pth".format(args.exp_id,args.pre_model_id,bestPreTrialNb)
        print("args.init_weights",args.init_weights)

        trainer = MetaTrainer(args)
        trainer.train(trial)

        args.eval_weights = "../models/{}/meta_{}_trial{}_max_acc.pth".format(args.exp_id,args.model_id,trial.number)
        val = trainer.eval()

    elif args.phase == "pre_train":
        trainer = PreTrainer(args)
        val = trainer.train()

    else:
        raise ValueError("Unkown phase",args.phase)

    return val

def findBestTrial(args,pre):

    id = args.pre_model_id if pre else args.model_id

    con = sqlite3.connect("../results/{}/{}_hypSearch.db".format(args.exp_id,id))
    curr = con.cursor()

    curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1')
    query_res = curr.fetchall()

    query_res = list(filter(lambda x:not x[1] is None,query_res))

    trialIds = [id_value[0] for id_value in query_res]
    values = [id_value[1] for id_value in query_res]

    trialIds = trialIds[:args.optuna_trial_nb]
    values = values[:args.optuna_trial_nb]

    bestTrial = trialIds[np.array(values).argmax()]

    curr.execute("SELECT param_value FROM trial_params WHERE trial_id == {} and param_name == 'nb_parts' ".format(bestTrial))
    query_res = curr.fetchall()
    best_part_nb = int(query_res[0][0])

    return bestTrial,best_part_nb

def writeConfigFile(args,filePath):
    """ Writes a config file containing all the arguments and their values"""

    config = configparser.ConfigParser()
    config.add_section('default')

    for k, v in  vars(args).items():
        config.set('default', k, str(v))

    with open(filePath, 'w') as f:
        config.write(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='ResNet', choices=['ResNet']) # The network architecture
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['miniImageNet', 'tieredImageNet', 'FC100']) # Dataset
    parser.add_argument('--phase', type=str, default='meta_train', choices=['pre_train', 'meta_train']) # Phase
    parser.add_argument('--seed', type=int, default=0) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', default='1') # GPU id
    parser.add_argument('--dataset_dir', type=str, default='./data/mini/') # Dataset folder

    # Parameters for meta-train phase
    parser.add_argument('--max_epoch', type=int, default=100) # Epoch number for meta-train phase
    parser.add_argument('--num_batch', type=int, default=100) # The number for different tasks used for meta-train
    parser.add_argument('--shot', type=int, default=1) # Shot number, how many samples for one class in a task
    parser.add_argument('--way', type=int, default=5) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=15) # The number of training samples for each class in a task
    parser.add_argument('--val_query', type=int, default=15) # The number of test samples for each class in a task
    parser.add_argument('--meta_lr1', type=float, default=0.0001) # Learning rate for SS weights
    parser.add_argument('--meta_lr2', type=float, default=0.001) # Learning rate for FC weights
    parser.add_argument('--base_lr', type=float, default=0.01) # Learning rate for the inner loop
    parser.add_argument('--update_step', type=int, default=50) # The number of updates for the inner loop
    parser.add_argument('--step_size', type=int, default=10) # The number of epochs to reduce the meta learning rates
    parser.add_argument('--gamma', type=float, default=0.5) # Gamma for the meta-train learning rate decay
    parser.add_argument('--init_weights', type=str, default=None) # The pre-trained weights for meta-train phase
    parser.add_argument('--eval_weights', type=str, default=None) # The meta-trained weights for meta-eval phase
    parser.add_argument('--meta_label', type=str, default='exp1') # Additional label for meta-train

    # Parameters for pretain phase
    parser.add_argument('--pre_max_epoch', type=int, default=40) # Epoch number for pre-train phase
    parser.add_argument('--pre_batch_size', type=int, default=128) # Batch size for pre-train phase
    parser.add_argument('--pre_lr', type=float, default=0.1) # Learning rate for pre-train phase
    parser.add_argument('--pre_gamma', type=float, default=0.2) # Gamma for the pre-train learning rate decay
    parser.add_argument('--pre_step_size', type=int, default=30) # The number of epochs to reduce the pre-train learning rate
    parser.add_argument('--pre_custom_momentum', type=float, default=0.9) # Momentum for the optimizer during pre-train
    parser.add_argument('--pre_custom_weight_decay', type=float, default=0.0005) # Weight decay for the optimizer during pre-train
    parser.add_argument('--nb_parts', type=int, default=3) #
    parser.add_argument('--max_batch_size', type=int, default=256)

    #
    parser.add_argument('--exp_id', type=str,default="default")
    parser.add_argument('--model_id', type=str,default="default")
    parser.add_argument('--pre_model_id', type=str,default="pre_default")
    parser.add_argument('--optuna_trial_nb', type=int,default=25)
    parser.add_argument('--num_workers', type=int,default=6)

    # Set and print the parameters
    args = parser.parse_args()
    pprint(vars(args))

    # Set the GPU id
    set_gpu(args.gpu)

    # Set manual seed for PyTorch
    if args.seed==0:
        print ('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print ('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not (os.path.exists("../results/{}".format(args.exp_id))):
        os.makedirs("../results/{}".format(args.exp_id))
    if not (os.path.exists("../models/{}".format(args.exp_id))):
        os.makedirs("../models/{}".format(args.exp_id))

    writeConfigFile(args,"../models/{}/{}.ini".format(args.exp_id, args.model_id))

    def objective(trial):
        return run(args,trial=trial)

    study = optuna.create_study(direction="maximize",\
                                storage="sqlite:///../results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id), \
                                study_name=args.model_id,load_if_exists=True)

    con = sqlite3.connect("../results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id))
    curr = con.cursor()

    failedTrials = 0
    for elem in curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1').fetchall():
        if elem[1] is None:
            failedTrials += 1

    trialsAlreadyDone = len(curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1').fetchall())

    if trialsAlreadyDone-failedTrials < args.optuna_trial_nb:

        studyDone = False
        while not studyDone:
            try:
                print("N trials left",args.optuna_trial_nb-trialsAlreadyDone+failedTrials)
                study.optimize(objective,n_trials=args.optuna_trial_nb-trialsAlreadyDone+failedTrials)
                studyDone = True
            except RuntimeError as e:
                print("------- Max batch size was {} -------".format(args.max_batch_size))
                if str(e).find("CUDA out of memory.") != -1:
                    gc.collect()
                    torch.cuda.empty_cache()
                    args.max_batch_size -= 5
                else:
                    raise RuntimeError(e)

    if args.phase == "meta_train":
        bestTrialId,args.nb_parts = findBestTrial(args,pre=False)
        args.eval_weights = "../models/{}/meta_{}_trial{}_max_acc.pth".format(args.exp_id,args.model_id,bestTrialId-1)

        copyfile(args.eval_weights, args.eval_weights.replace("_trial{}".format(bestTrialId-1),""))

        trainer = MetaTrainer(args)
        trainer.eval()
