
python3 processResults.py --model_id firstModel_bcnn --nrows 7 --plot_id main  \
                          --high_res_id firstModel_dist --gradcam_id firstModel_long_fix \
                          --batch_inds_to_plot 2 16 19   13 13 13   25 22 12 \
                          --classes_to_plot    4 4 0    1 1 1   0 4 1 \
                          --img_to_plot        1 0 0    1 2 3   2 1 4

python3 processResults.py --model_id firstModel_bcnn --nrows 7 --plot_id teaser --batch_inds_to_plot 2 16 \
                          --classes_to_plot 4 4 --high_res_id firstModel_dist --nb_per_class 5 --gradcam_id firstModel_long_fix \
                          --img_to_plot 1 0

python3 processResults.py --model_id firstModel_bcnn --nrows 7 --plot_id all --high_res_id firstModel_dist --nb_per_class 5 --gradcam_id firstModel_long_fix


################### ALL #########################
python3 processResults.py --model_id firstModel_dist --nrows 7 --plot_id all --high_res_id firstModel_dist
python3 processResults.py --model_id firstModel_dist --nrows 7 --plot_id all_val --high_res_id firstModel_dist --test_on_val
