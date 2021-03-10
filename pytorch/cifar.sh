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
  "pre_long_fc100_merged")
    python3 main.py --dataset_dir datasets/FC100/ --exp_id fc100 --model_id pre_firstModel_long_merged     --phase pre_train --optuna_trial_nb 25 --repvec_merge True
    ;;
  "pre_long_fc100")
    python3 main.py --dataset_dir datasets/FC100/ --exp_id fc100 --model_id pre_firstModel_long     --phase pre_train --optuna_trial_nb 25
    ;;
  "pre_longer_fc100")
    python3 main.py --dataset_dir datasets/FC100/ --exp_id fc100 --model_id pre_firstModel_longer     --phase pre_train --optuna_trial_nb 50
    ;;
  "long_fc100")
    python3 main.py --dataset_dir datasets/FC100/ --exp_id fc100 --model_id firstModel_long     --phase meta_train --pre_model_id pre_firstModel_long --max_epoch 100 --optuna_trial_nb 55
    ;;
  "long_fc100_fix")
    python3 main.py --dataset_dir datasets/FC100/ --exp_id fc100 --model_id firstModel_long_fix     --phase meta_train --pre_model_id pre_firstModel_long --max_epoch 100 --optuna_trial_nb 55 --fix_trial_id True
    ;;
  "pre_noRV")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id pre_firstModel_noRV     --phase pre_train --optuna_trial_nb 10 --rep_vec False
    ;;
  "pre_dist")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id pre_firstModel_dist     --phase pre_train --optuna_trial_nb 1 --distill_id pre_firstModel --max_batch_size 150 --val_query 7
    ;;
  "dist")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_dist     --phase meta_train --pre_model_id pre_firstModel_dist --max_epoch 100 --optuna_trial_nb 10 --distill_id firstModel_long_fix --distill_id_pre pre_firstModel  --max_batch_size 150 --val_query 7 --fix_trial_id True
    ;;
  "pre_dist_fix")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id pre_firstModel_dist_fix     --phase pre_train --optuna_trial_nb 5 --distill_id pre_firstModel --max_batch_size 150 --val_query 7
    ;;
  "dist_fix")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_dist_fix     --phase meta_train --pre_model_id pre_firstModel_dist --max_epoch 100 --optuna_trial_nb 10 --distill_id firstModel_long_fix --distill_id_pre pre_firstModel  --fix_trial_id True  --max_batch_size 150 --val_query 7
    ;;
  "dist_fix_best")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_dist_fix --phase meta_train --val_query 5 --fix_trial_id True --best True --high_res True --trial_id 5
    ;;
  "dist_best")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_dist     --phase meta_train --max_epoch 80 --optuna_trial_nb 10 --max_batch_size 150 --val_query 5 --fix_trial_id True \
                     --best True --high_res True --trial_id 139
    ;;
  "long_noRV")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_noRV     --phase meta_train --pre_model_id pre_firstModel_noRV --max_epoch 100 --optuna_trial_nb 10 --rep_vec False --fix_trial_id True
    ;;
  "long")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_long     --phase meta_train --pre_model_id pre_firstModel --max_epoch 100 --optuna_trial_nb 55
    ;;
  "long_fix")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_long_fix     --phase meta_train --pre_model_id pre_firstModel --max_epoch 100 --optuna_trial_nb 10 --fix_trial_id True
    ;;
  "long_fix_HT")
    python3 main.py --dataset_dir datasets/cifar-FS/ --exp_id cifar --model_id firstModel_long_fix_HT     --phase meta_train --pre_model_id pre_firstModel --max_epoch 100 --optuna_trial_nb 50 --hard_tasks True --fix_trial_id True
    ;;
  "*")
    echo "no such model"
    ;;
esac

# best distill params --base_lr 0.00305092882996413 --gamma 0.9 --kl_interp 0.4 --kl_temp 16.0 --meta_lr1 3.84491256874587e-05 --meta_lr2 0.00339547011477872 --step_size 17.0 --update_step 64.0
