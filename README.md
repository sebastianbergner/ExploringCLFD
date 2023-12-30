# Effect of Optimizer, Initializer, and Architecture of Hypernetworks on Continual Learning from Demonstration

In *continual learning from demonstration* (CLfD), a robot learns a sequence of real-world motion skills continually from human demonstrations. Recently, hypernetworks have been successful in solving this problem. In our current work, we perform an exploratory study of the effects of different optimizers, initializers, and network architectures on the continual learning performance of hypernetworks for CLfD. Our results show that adaptive learning rate optimizers work well, but initializers specially designed for hypernetworks offer no advantages for CLfD. We also show that hypernetworks that are capable of *stable* trajectory predictions are robust to different network architectures. We use the [*RoboTasks9*](https://arxiv.org/abs/2311.03600) real-world LfD benchmark (shown below) for evaluations.

![all_tasks_rt9](https://github.com/sebastianbergner/ExploringCLFD/assets/10401716/fd074d1c-3ded-4021-b454-b19d18e94561)


## Table of Contents
- [Setup](#Setup)
- [Usage](#Usage)
- [Experiment Hyperparameters](#Hyperparameters)
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

## Hyperparameters

![hn_explore_hparam](https://github.com/sebastianbergner/ExploringCLFD/assets/10401716/f6335e57-cb51-498e-96a1-4586701ac53e)


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
