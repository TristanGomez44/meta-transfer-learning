python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id pre_firstModel --phase pre_train
python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel     --phase meta_train --pre_model_id pre_firstModel
