case $1 in
  "")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel     --phase meta_train --pre_model_id pre_firstModel --optuna_trial_nb 100
    ;;
  "pre_best")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id pre_firstModel   --best True   --phase pre_train --optuna_trial_nb 50 --gpu 0,1,2,3
    ;;
  "pre_long")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id pre_firstModel_long     --phase pre_train --optuna_trial_nb 50
    ;;
  "pre_noRV")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id pre_firstModel_noRV     --phase pre_train --optuna_trial_nb 25 --rep_vec False
    ;;
  "pre_dist")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id pre_firstModel_dist     --phase pre_train --optuna_trial_nb 25 --distill_id pre_firstModel
    ;;
  "long_noRV")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_noRV     --phase meta_train --pre_model_id pre_firstModel_noRV --max_epoch 100 --optuna_trial_nb 25 --rep_vec False --fix_trial_id True
    ;;
  "long")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_long     --phase meta_train --pre_model_id pre_firstModel --max_epoch 100 --optuna_trial_nb 55
    ;;
  "long_fix")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_long_fix     --phase meta_train --pre_model_id pre_firstModel --max_epoch 100 --optuna_trial_nb 50 --fix_trial_id True
    ;;
  "long_fix_HT")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_long_fix_HT     --phase meta_train --pre_model_id pre_firstModel --max_epoch 100 --optuna_trial_nb 50 --hard_tasks True --fix_trial_id True
    ;;
  "*")
    echo "no such model"
    ;;
esac
