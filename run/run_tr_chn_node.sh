#! /bin/bash 
python tr_chn_node.py \
--data_dir datasets/robottasks/pos_ori \
--num_iter 10 \
--tsub 20 \
--replicate_num 0 \
--lr 0.00005 \
--tnet_dim 6 \
--tnet_arch 1000,1000,1015 \
--tnet_act elu \
--init kaiming \
--optimizer Adam \
--hnet_arch 300,300,300 \
--task_emb_dim 256 \
--chunk_emb_dim 256 \
--chunk_dim 8192 \
--explicit_time 1 \
--chn_type complex \
--zero_center 0 \
--int_method euler \
--dropout -1.0 \
--beta 0.005 \
--lr_change_iter -1 \
--data_class RobotTasksPositionOrientation \
--eval_during_train 0 \
--seed 100 \
--seq_file datasets/robottasks/robottasks_pos_ori_sequence_all.txt \
--log_dir logs/ \
--description robottasks_pos_ori_test \
--data_type np \
--tangent_vec_scale 5.0
