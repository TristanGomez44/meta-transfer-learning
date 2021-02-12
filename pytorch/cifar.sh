case $1 in
  "")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel     --phase meta_train --pre_model_id pre_firstModel
    ;;
  "long")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_long     --phase meta_train --pre_model_id pre_firstModel --max_epoch 100
    ;;
  "*")
    echo "no such model"
    ;;
esac
