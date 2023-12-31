#! /bin/bash 
python tr_chn_lsddm.py \
--data_dir datasets/robottasks/pos_ori \
--num_iter 1 \
--tsub 20 \
--replicate_num 0 \
--lr 0.00005 \
--tnet_dim 6 \
--fhat_layers 3 \
--tnet_act elu \
--init kaiming \
--optimizer Adam \
--hnet_arch 300,300,300 \
--task_emb_dim 256 \
--chunk_emb_dim 256 \
--chunk_dim 8192 \
--explicit_time 1 \
--chn_type complex \
--beta 0.005 \
--lr_change_iter -1 \
--lsddm_a 0.5 \
--lsddm_projfn PSD-REHU \
--lsddm_projfn_eps 0.0001 \
--lsddm_smooth_v 0 \
--lsddm_hp 100 \
--lsddm_h 1000 \
--lsddm_rehu 0.01 \
--dummy_run 0 \
--data_class RobotTasksPositionOrientation \
--eval_during_train 0 \
--seed 100 \
--seq_file datasets/robottasks/robottasks_pos_ori_sequence_all.txt \
--log_dir logs/ \
--description robottasks_pos_ori_test \
--tangent_vec_scale 5.0
