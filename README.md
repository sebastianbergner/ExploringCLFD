# Exploring Aspects in Continual Learning from Demonstration of Robotic Skills

Recently, hypernetworks have been proposed to counter the catastrophic forgetting effect. Hypernetworks are neural networks that produce another neural network as its output, and it has been shown that they perform better than other methods for continual learning on standard image benchmark datasets, as well as on tasks in robotics. Robotics is an excellent test bed for continual learning algorithms, since a robot should be able to learn and remember new tasks throughout its lifetime without forgetting previously learned skills. However, the previous works with hypernetworks do not fully explore all aspects of hypernetworks which can be exploited to attain even better performance. The goal of this paper is to fill this gap and explore the effect of different factors on continually learning hypernetworks in a robotics scenario. These factors may include optimization algorithms, network initialization schemes, and different network architectures.

## Table of Contents
- [Introduction](#Exploring Aspects in Continual Learning from Demonstration of Robotic Skills)
- [Setup](#Setup)
- [Usage](#Usage)
- [Bibliobgraphy](#Bibliobgraphy)

## Setup

Clone this repository.

```bash
git clone https://github.com/sebastianbergner/ExploringCLFD.git
```

Set up a new virtual Python environment, activate it and install the required packages.

```bash
python3 -m venv CLFDvenv
source CLFDvenv/bin/activate
pip install -r requirements.txt
```

## Usage

For ease of execution we added a few bash scripts, located in `run`, with example commands.

```bash
python tr_hn_node.py \
--data_dir datasets/robottasks/pos_ori \
--num_iter 10 \
--tsub 20 \
--replicate_num 0 \
--lr 0.00005 \
--tnet_dim 6 \
--tnet_arch 100,100,150 \
--tnet_act elu \
--init principled_weight_init_uniform \
--hnet_arch 300,300,300 \
--task_emb_dim 256 \
--explicit_time 1 \
--int_method euler \
--optimizer Adam \
--dropout -1.0 \
--beta 0.005 \
--data_class RobotTasksPositionOrientation \
--eval_during_train 0 \
--seed 100 \
--seq_file datasets/robottasks/robottasks_pos_ori_sequence_all.txt \
--log_dir logs/ \
--description test_tr_hn_node \
--data_type np \
--tangent_vec_scale 5.0
```

The parameters explained are below.

- data_dir         Specifies the location of the dataset.
- num_iter         Number of iterations (e.g., 40,000 for the full run, 10 for demonstration).
- tsub             Length of trajectory subsequences for training (required for LASA dataset).
- replicate_num    Number of times the final point of the trajectories should be replicated for training.
- lr               Learning rate.
- tnet_dim         Dimension of target network input and output.
- tnet_arch        Target network architecture.
- tnet_act         Activation function of the target network (default: elu).
- init             Initialization method for the hypernetwork (kaiming/xavier/principled_weight_init_uniform/principled_weight_init_normal).
- hnet_arch        Hypernetwork architecture.
- task_emb_dim     Dimension of the task embedding vector.
- explicit_time    1 if an additional time input should be used in the target network (default: 1).
- int_method       Integration method (euler/dopri5).
- optimizer        Optimizer for backpropagation adjustment (Adam, RMSProp, SGD).
- dropout          Dropout rate. -1.0 means no dropout.
- data_class       Data class specification (e.g., RobotTasksPositionOrientation).
- eval_during_train Flag for evaluating during training (0 or 1).
- seed             Random seed.
- seq_file         File specifying the sequences. Located in the dataset directories.
- log_dir          Directory for logs.
- description      Description comment of the run.
- data_type        Data type np for RoboTasks9, mat for LASA.
- tangent_vec_scale Scaling of tangent vectors for learning oritentation (default 5.0).


## Bibliobgraphy

The below listed citations are the most relevant ones, all can be found in our paper.

- Sayantan Auddy, Jakob Hollenstein, Matteo Saveriano, Antonio Rodríguez-Sánchez, and Justus Piater. Continual learning from demonstration of robotics skills. Robotics and Autonomous Systems, 165:104427, 2023a. [Paper](https://arxiv.org/abs/2202.06843)

- Sayantan Auddy, Jakob Hollenstein, Matteo Saveriano, Antonio Rodríguez-Sánchez, and Justus Piater. Scalable and efficient continual learning from demonstration via hypernetwork generated stable dynamics model, 2023b. [Paper](https://arxiv.org/abs/2311.03600)

- Oscar Chang, Lampros Flokas, and Hod Lipson. Principled weight initialization for hypernetworks. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. [Paper](https://openreview.net/forum?id=H1lma24tPB)

- Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations, 2019. [Paper](https://arxiv.org/abs/1806.07366)

- Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Yee Whye Teh and Mike Titterington, editors, Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pages 249–256, Chia Laguna Resort, Sardinia, Italy, 13–15 May 2010. PMLR.  [Paper](https://proceedings.mlr.press/v9/glorot10a.html)

- David Ha, Andrew Dai, and Quoc V. Le. Hypernetworks, 2016. URL [Paper](https://arxiv.org/abs/1609.09106)

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pages 1026–1034, 2015. [Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)

- Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. [Paper](https://arxiv.org/abs/1412.6980)

- J. Zico Kolter and Gaurav Manek. Learning stable deep dynamics models. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. [Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/0a4bbceda17a6253386bc9eb45240e25-Paper.pdf)

- Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude, 2012. [Lecture Slides](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

- Johannes von Oswald, Christian Henning, João Sacramento, and Benjamin F. Grewe. Continual learning with hypernetworks. CoRR, abs/1906.00695, 2019. URL [Paper](http://arxiv.org/abs/1906.00695)
