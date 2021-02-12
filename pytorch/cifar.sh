case $1 in
  "")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel     --phase meta_train --pre_model_id pre_firstModel --optuna_trial_nb 50
    ;;
  "pre_long")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id pre_firstModel_long     --phase pre_train --optuna_trial_nb 50
    ;;
  "long")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_long     --phase meta_train --pre_model_id pre_firstModel --max_epoch 100 --optuna_trial_nb 50
    ;;
  "*")
    echo "no such model"
    ;;
esac
