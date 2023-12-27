#! /bin/bash 
python tr_hn_lsddm.py \
--data_dir datasets/robottasks/pos_ori \
--num_iter 100 \
--tsub 20 \
--replicate_num 0 \
--lr 0.00005 \
--tnet_dim 6 \
--init kaiming \
--optimizer Adam \
--explicit_time 1 \
--hnet_arch 300,300,300 \
--task_emb_dim 256 \
--beta 0.005 \
--lr_change_iter -1 \
--lsddm_a 0.5 \
--lsddm_projfn PSD-REHU \
--lsddm_projfn_eps 0.0001 \
--lsddm_smooth_v 0 \
--lsddm_hp 101 \
--lsddm_h 100 \
--lsddm_rehu 0.01 \
--fhat_layers 3 \
--dummy_run 0 \
--data_class RobotTasksPositionOrientation \
--eval_during_train 0 \
--seed 100 \
--seq_file datasets/robottasks/robottasks_pos_ori_sequence_all.txt \
--log_dir logs/ \
--tb_log_dir runs/ \
--description test_tr_hn_lsddm \
--tangent_vec_scale 5.0 
