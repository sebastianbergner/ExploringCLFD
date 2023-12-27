import os
from tqdm import trange
from argparse import ArgumentParser
import logging
import numpy as np
from copy import deepcopy
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from hypernet_explore.train.utils import check_cuda, set_seed, get_sequence, get_beta_for_tasks
from hypernet_explore.model.hypernetwork import ChunkedHyperNetwork, ChunkedTaskEmbHypernetwork, str_to_ints, \
    get_current_targets, calc_delta_theta, calc_fix_target_reg, str_to_init, str_to_optim
from hypernet_explore.model.node import NODE
from hypernet_explore.model.lsddm_hn import configure as configure
from hypernet_explore.model.lsddm_hn_t import configure as configure_t
from hypernet_explore.data.lasa import LASAExtended
from hypernet_explore.data.helloworld import HelloWorldExtended
from hypernet_explore.data.robottasks import RobotTasksPositionOrientation
from hypernet_explore.data.utils import get_minibatch_extended as get_minibatch
from hypernet_explore.metrics.traj_metrics import mean_swept_error, mean_frechet_error_fast as mean_frechet_error, dtw_distance_fast as dtw_distance
from hypernet_explore.metrics.ori_metrics import quat_traj_distance
from hypernet_explore.logging.utils import custom_logging_setup, write_dict, read_dict

#TODO Remove later
# Warning is a PyTorch bug
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


def parse_args(return_parser=False):
    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True, help='Location of dataset')
    parser.add_argument('--num_iter', type=int, required=True, help='Number of training iterations')
    parser.add_argument('--tsub', type=int, default=20, help='Length of trajectory subsequences for training')
    parser.add_argument('--replicate_num', type=int, default=0, help='Number of times the final point of the trajectories should be replicated for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--tnet_dim', type=int, default=2, help='Dimension of target network input and output')
    parser.add_argument('--fhat_layers', type=int, required=True, help='Number of hidden layers in the fhat of target network')
    parser.add_argument('--tnet_act', type=str, default='elu', help='Target network activation function')
    parser.add_argument('--init', type=str, default='kaiming', help='Initialization function')
    parser.add_argument('--init_bias_pwi', type=int, default=0, help='Use PWI bias initialization (only used with PWI)')
    parser.add_argument('--optimizer', type=str, default='adam', help='Type of optimizer to use (Adam, AdamW, RMSProp or SGD)')
    parser.add_argument('--hnet_arch', type=str, default='200,200,200', help='Hidden layer units of the hypernetwork')
    parser.add_argument('--task_emb_dim', type=int, default=5, help='Dimension of the task embedding vector')
    parser.add_argument('--chunk_emb_dim', type=int, default=5, help='Dimension of the each chunk embedding vector (input to the HN)')
    parser.add_argument('--chunk_dim', type=int, default=1000, help='Dimension of the output of the chunked HN (these chunks are tiled together to create the final target network)')
    parser.add_argument('--explicit_time', type=int, default=0, help='1: Use time as an explicit network input, 1: Do not use time')
    parser.add_argument('--chn_type', type=str, default='complex', help='complex: CHN with a separate regularized chunk embedding, complex: CHN with a chunked task embedding and no separate chunk embedding')
    parser.add_argument('--beta', type=float, default=5e-3, help='Regularization strength')
    parser.add_argument('--beta_decay', type=float, default=1.0, help='Multiplicative factor for beta (for each task: beta *= beta_decay')

    parser.add_argument('--lr_change_iter', type=int, default=-1, help='-1 or 0: No LR scheduler, >0: Number of iterations after which initial LR is divided by 10')

    # Scaling term for tangent vectors for learning orientation
    parser.add_argument('--tangent_vec_scale', type=float, default=1.0, help='Tangent vector scaling term')

    parser.add_argument('--lsddm_a', type=float, default=0.5)
    parser.add_argument('--lsddm_projfn', type=str, default='PSD-REHU', help='LSDDM projection function')
    parser.add_argument('--lsddm_projfn_eps', type=float, default=0.0001)
    parser.add_argument('--lsddm_smooth_v', type=int, default=0)
    parser.add_argument('--lsddm_hp', type=int, default=60)
    parser.add_argument('--lsddm_h', type=int, default=100)
    parser.add_argument('--lsddm_rehu', type=float, default=0.01)

    parser.add_argument('--dummy_run', type=int, default=0, help='1: Dummy run, no actual evaluation, 0: Actual training run')

    parser.add_argument('--data_class', type=str, required=True, help='Dataset class for training')
    parser.add_argument('--eval_during_train', type=int, default=0, help='0: net for a task is evaluated immediately after training, 1: eval for all nets is done after training of all tasks')
    parser.add_argument('--seed', type=int, required=True, help='Seed for reproducability')
    parser.add_argument('--seq_file', type=str, required=True, help='Name of file containing sequence of demonstration files')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Main directory for saving logs')
    parser.add_argument('--description', type=str, required=True, help='String identifier for experiment')

    if return_parser:
        # This is used by the slurm creator script
        # When running this script directly, this has no effect
        return parser
    else:
        args = parser.parse_args()
        return args


def train_task(args, task_id, hnet, tnet, node, beta, device, param_names, pbar=trange, writer=None, save_dir=""):

    starttime = time.time()

    filenames = get_sequence(args.seq_file)

    assert 0.0<=beta<=1.0, f'Invalid beta: {beta}'

    dataset = None
    if args.data_class == 'LASA':
        datafile = os.path.join(args.data_dir, filenames[task_id])
        dataset = LASAExtended(datafile, seq_len=args.tsub, norm=True, device=device)

        # Goal position at origin
        dataset.zero_center()

    elif args.data_class == 'HelloWorld':
        dataset = HelloWorldExtended(data_dir=args.data_dir, filename=filenames[task_id], device=device)

        # Goal position at origin
        dataset.zero_center()
    elif args.data_class == 'RobotTasksPositionOrientation':
        dataset = RobotTasksPositionOrientation(data_dir=args.data_dir, datafile=filenames[task_id], device=device, scale=args.tangent_vec_scale)

        # Goal position at origin
        dataset.zero_center()
    else:
        raise NotImplementedError(f'Unknown dataset class {args.data_class}')

    node.set_target_network(tnet)

    tnet.train()
    hnet.train()
    node.train()

    # Create a new task embedding for this task
    hnet.gen_new_task_emb()

    tnet = tnet.to(device)
    hnet = hnet.to(device)
    node = node.to(device)

    # Get the parameters generated by the hnet for all tasks
    # preceeding the current task_id. This will be used for 
    # calculating the regularized targets.
    if args.beta > 0:
        targets = get_current_targets(task_id, hnet)

    # Trainable weights and biases of the hnet
    regularized_params = list(hnet.theta)
               
    # For optimizing the weights and biases of the hnet
    #theta_optimizer = optim.Adam(regularized_params, lr=args.lr)
    optim_fn = str_to_optim(args.optimizer)
    theta_optimizer = optim_fn(regularized_params, lr=args.lr)

    # Apply learning scheduler if needed
    if args.lr_change_iter > 0:
        theta_lambda = lambda epoch: 1.0 if (epoch < args.lr_change_iter) else 0.1
        theta_scheduler = LambdaLR(theta_optimizer, lr_lambda=theta_lambda)

    # For optimizing the task embedding for the current task.
    # We only optimize the task embedding corresponding to the current task,
    # the remaining ones stay constant.
    #emb_optimizer = optim.Adam([hnet.get_task_emb(task_id)], lr=args.lr)
    emb_optimizer = optim_fn([hnet.get_task_emb(task_id)], lr=args.lr)

    # Whether the regularizer will be computed during training?
    calc_reg = task_id > 0 and args.beta > 0

    best_loss = np.inf
    best_iter = 0

    hnet.init_batch_entropy_new_task()

    # Start training iterations
    for iteration in trange(args.num_iter):

        # Set flag for storing batch entropy
        hnet.compute_batch_entropy = True

        ### Train theta and task embedding
        theta_optimizer.zero_grad()
        emb_optimizer.zero_grad()

        # Generate parameters of the target network for the current task
        weights = hnet.forward(task_id)

        # Set the weights of the target network
        tnet.set_weights(weights, param_names)

        # Set the target network in the NODE
        node.set_target_network(tnet)

        # Train using the translated trajectory (with goal at the origin)
        t, y_all = get_minibatch(dataset.t[0], dataset.pos_goal_origin, nsub=None, tsub=args.tsub, dtype=torch.float)

        # We use the timesteps associated with the first sequence
        # Starting points
        y_start = y_all[:,0].float()
        y_start.requires_grad = True

        # Predicted trajectories - forward simulation
        y_hat = node(t.float(), y_start) 
        
        # MSE
        loss = ((y_hat-y_all)**2).mean()

        # Log the loss in tensorboard
        if writer is not None:
            writer.add_scalar(f'task_loss/task_{task_id}', loss.item(), iteration)

        # Calling loss_task.backward computes the gradients w.r.t. the loss for the 
        # current task. 
        # Here we keep dtheta fixed, hence we do not need to create a graph of the derivatives
        # and so create_graph=False
        # The graph needs to be preserved only when the regulation loss is to be backpropagated
        # and so retain_graph is True only when calc_reg is True
        loss.backward(retain_graph=calc_reg, create_graph=False)

        # The task embedding is only trained on the task-specific loss.
        # Note, the gradients accumulated so far are from "loss_task".
        emb_optimizer.step()

        # Initialize the regularization loss
        loss_reg = 0

        # Initialize dTheta, the candidate change in the hnet parameters
        dTheta = None

        # Unset flag for storing batch entropy
        # We want to compute the batch entropy only during the forward pass to generate the target network
        # The forward passes during regularization should not be used for computing the batch entropy
        hnet.compute_batch_entropy = False

        if calc_reg:

            # Find out the candidate change (dTheta) in trainable parameters (theta) of the hnet
            # This function just computes the change (dTheta), but does not apply it
            dTheta = calc_delta_theta(theta_optimizer,
                                      False, 
                                      lr=args.lr,
                                      detach_dt=True)

            # Calculate the regularization loss using dTheta
            # This implements the second part of equation 2
            loss_reg = calc_fix_target_reg(hnet, 
                                           task_id,
                                           targets=targets, 
                                           dTheta=dTheta)

            # Multiply the regularization loss with the scaling factor
            # loss_reg *= args.beta
            loss_reg *= beta  # We use the beta supplied explicitly (to nable beta decay)   

            # Log the loss in tensorboard
            if writer is not None:
                writer.add_scalar(f'reg_loss/task_{task_id}', loss_reg.item(), iteration)

            # Backpropagate the regularization loss
            loss_reg.backward()

        # Update the hnet params using the current task loss and the regularization loss
        theta_optimizer.step()

        if args.lr_change_iter > 0:
            theta_scheduler.step()

        if loss.item() <= best_loss:
            best_hnet = deepcopy(hnet)
            best_loss = loss.item()
            best_iter = int(iteration)

    endtime = time.time()
    duration = endtime - starttime

    # Save the computed batch entropies to disk
    hnet.store_batch_entropy(save_dir, task_id)

    return best_hnet, duration, best_loss, best_iter

def eval_task(args, eval_task_id, hnet, tnet, node, device, param_names, train_task_id, writer=None):

    hnet.eval()
    tnet.eval()
    node.eval()

    hnet.compute_batch_entropy = False

    tnet = tnet.to(device)
    hnet = hnet.to(device)
    node = node.to(device)

    filenames = get_sequence(args.seq_file)

    data = None
    if args.data_class == 'LASA':
        datafile = os.path.join(args.data_dir, filenames[eval_task_id])
        dataset = LASAExtended(datafile, seq_len=args.tsub, norm=True, device=device)

        # Goal position at origin
        dataset.zero_center()
    elif args.data_class == 'HelloWorld':
        dataset = HelloWorldExtended(data_dir=args.data_dir, filename=filenames[eval_task_id], device=device)

        # Goal position at origin
        dataset.zero_center()
    elif args.data_class == 'RobotTasksPositionOrientation':
        dataset = RobotTasksPositionOrientation(data_dir=args.data_dir, datafile=filenames[eval_task_id], device=device, scale=args.tangent_vec_scale)

        # Goal position at origin
        dataset.zero_center()
    else:
        raise NotImplementedError(f'Unknown dataset class {args.data_class}')

    # Generate parameters of the target network for the current task
    weights = hnet.forward(eval_task_id)

    # Set the weights of the target network
    tnet.set_weights(weights, param_names)

    # Set the target network in the NODE
    node.set_target_network(tnet)
    node = node.float()
    node.eval()

    # The time steps
    t = dataset.t[0].float()

    # The starting position 
    # (n,d-dimensional, where n is the num of demos and 
    # d is the dimension of each point)
    y_start = dataset.pos_goal_origin[:,0]
    y_start = y_start.float()
    y_start.requires_grad = True    

    # The entire demonstration trajectory
    y_all = dataset.pos.float()

    # The predicted trajectory is computed in a piecemeal fashion
    # Predicted trajectory
    t_step = 20
    t_start = 0
    t_end = t_start + t_step
    y_start = y_start
    y_hats = list()
    i = 0
    
    while t_end <= y_all.shape[1]:
        i += 1
        y_hat = node(t[t_start:t_end], y_start)
        y_hats.append(y_hat)
        y_start = y_hat[:,-1,:].detach().clone()
        y_start.requires_grad = True
        t_start = t_end
        t_end = t_start + t_step

    y_hat_zeroed = torch.cat(y_hats, 1)
    y_hat = dataset.unzero_center(y_hat_zeroed)
    y_hat_np = y_hat.cpu().detach().numpy()

    # Compute trajectory metrics
    y_all_np = y_all.cpu().detach().numpy()

    # De-normalize the data before computing trajectories
    y_all_np_denorm = dataset.denormalize(y_all_np)
    y_hat_np_denorm = dataset.denormalize(y_hat_np)

    if args.data_class == 'RobotTasksPositionOrientation':
        # Separate the position and rotation vectors
        # Predictions
        position_hat_np = y_hat_np_denorm[:,:,:3]
        rotation_hat_np = y_hat_np_denorm[:,:,3:]
        # Ground truth
        position_all_np = y_all_np_denorm[:,:,:3]
        rotation_all_np = y_all_np_denorm[:,:,3:]

        # Convert predicted rotation trajectory from tangent vectors to quaternions
        q_hat_np = dataset.from_tangent_plane(rotation_hat_np)

        # Compute metrics for position
        metric_swept_err, metric_swept_errs = mean_swept_error(position_all_np, position_hat_np)
        metric_frechet_err, metric_frechet_errs = mean_frechet_error(position_all_np, position_hat_np)
        metric_dtw_err, metric_dtw_errs = dtw_distance(position_all_np, position_hat_np)

        # Compute metrics for quaternion
        metric_quat_err, metric_quat_errs = quat_traj_distance(dataset.rotation_quat, q_hat_np)

        # Store the metrics
        eval_traj_metrics = {'swept': metric_swept_err, 
                             'frechet': metric_frechet_err, 
                             'dtw': metric_dtw_err,
                             'quat_error': metric_quat_err}
        # Convert np arrays to list so that these can be written to JSON
        eval_traj_metric_errors = {'swept': metric_swept_errs.tolist(), 
                                   'frechet': metric_frechet_errs.tolist(), 
                                   'dtw': metric_dtw_errs.tolist(),
                                   'quat_error': metric_quat_errs.tolist()}
    else:
        # Compute the error metric (array of metrics for each trajectory in the ground truth)
        if args.dummy_run == 0:
            metric_dtw_err, metric_dtw_errs = dtw_distance(y_all_np_denorm, y_hat_np_denorm)
            metric_frechet_err, metric_frechet_errs = mean_frechet_error(y_all_np_denorm, y_hat_np_denorm)
            metric_swept_err, metric_swept_errs = mean_swept_error(y_all_np_denorm, y_hat_np_denorm)
        elif args.dummy_run == 1:
            metric_dtw_err, metric_dtw_errs = 0, np.zeros(y_hat_np_denorm.shape[0])
            metric_frechet_err, metric_frechet_errs = 0, np.zeros(y_hat_np_denorm.shape[0])
            metric_swept_err, metric_swept_errs = 0, np.zeros(y_hat_np_denorm.shape[0])

        eval_traj_metrics = {'swept': metric_swept_err, 
                            'frechet': metric_frechet_err, 
                            'dtw': metric_dtw_err}

        # Store the metric errors
        # Convert np arrays to list so that these can be written to JSON
        eval_traj_metric_errors = {'swept': metric_swept_errs.tolist(), 
                                'frechet': metric_frechet_errs.tolist(), 
                                'dtw': metric_dtw_errs.tolist()}


    return eval_traj_metrics, eval_traj_metric_errors, node

def train_all(args):

    # Create logging folder and set up console logging
    save_dir, identifier = custom_logging_setup(args)

    # Tensorboard logging setup
    # writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tb', args.description, identifier))

    # Check if cuda is available
    cuda_available, device = check_cuda()
    logging.info(f'cuda_available: {cuda_available}')
        
    properties = {"latent_space_dim":args.tnet_dim,
                  "explicit_time": args.explicit_time,
                  "a":args.lsddm_a,
                  "projfn":args.lsddm_projfn,
                  "projfn_eps":args.lsddm_projfn_eps,
                  "smooth_v":args.lsddm_smooth_v,
                  "hp":args.lsddm_hp,
                  "h":args.lsddm_h,
                  "rehu":args.lsddm_rehu,
                  "device": device,
                  "fhat_layers": args.fhat_layers}

    # Create a LSDDM network 
    # Parameters are supplied during the forward pass of the hypernetwork
    # Load the network for the current task_id
    if args.explicit_time==1:
        properties["explicit_time"] = args.explicit_time
        target_network = configure_t(properties)
    elif args.explicit_time==0:
        target_network = configure(properties)

    target_network = target_network.to(device)

    # Shapes of the target network parameters
    param_names, param_shapes = target_network.get_param_shapes()

    # Create the chunked hypernetwork
    if args.chn_type == 'complex':
        hnet = ChunkedHyperNetwork(final_target_shapes=param_shapes,
                                   layers=str_to_ints(args.hnet_arch),
                                   init_fn=str_to_init(args.init),
                                   init_bias_pwi=args.init_bias_pwi,
                                   chunk_dim=args.chunk_dim,
                                   te_dim=args.task_emb_dim,
                                   ce_dim=args.chunk_emb_dim,
                                   verbose=True,
                                   device=device).to(device)
    elif args.chn_type == 'simple':
        hnet = ChunkedTaskEmbHypernetwork(target_shapes=param_shapes, 
                                          layers=str_to_ints(args.hnet_arch), 
                                          te_dim=args.task_emb_dim,
                                          init_fn=str_to_init(args.init),
                                          init_bias_pwi=args.init_bias_pwi,
                                          dropout_rate=-1, 
                                          device=device,
                                          chunk_dim=args.chunk_dim)
    else:
        raise NotImplementedError(f'Unknown chn_type: {args.chn_type}')
        
    # The NODE uses the target network as the RHS of its
    # differential equation
    # Apart from this, the NODE has no other trainable parameters
    node = NODE(target_network=target_network, method='euler', explicit_time=args.explicit_time).to(device)

    # Extract the list of demonstrations from the text file 
    # containing the sequence of demonstrations
    seq = get_sequence(args.seq_file)

    num_tasks = len(seq)

    eval_resuts=None

    # Get list of betas (one beta for each task)
    betas = get_beta_for_tasks(initial_beta=args.beta, beta_decay=args.beta_decay, num_tasks=num_tasks)
    logging.info(f'Betas: {betas}')

    for task_id in range(num_tasks):

        logging.info(f'#### Training started for task_id: {task_id} (task {task_id+1} out of {num_tasks}) ###')

        # Train on the current task_id
        hnet, duration, best_loss, best_iter = train_task(args, task_id, hnet, target_network, node, betas[task_id], device, param_names, pbar=trange, writer=None, save_dir=save_dir)

        logging.info(f'task_id: {task_id}, best_loss: {best_loss:.3E}, best_iter: {best_iter}')

        # At the end of every task store the latest hypernetwork
        logging.info('Saving models')
        torch.save(hnet, os.path.join(save_dir, 'models', f'hnet_{task_id}.pth'))

        if args.eval_during_train == 0:
            # Evaluate the latest network immediately after training
            # is complete for a task
            eval_resuts = eval_during_train(args, save_dir, task_id, eval_resuts, None)
        elif args.eval_during_train == 1:
            # Evaluation is done after training is finished for all tasks
            pass
        elif args.eval_during_train == 2:
            # No evaluation is performed, this is a trail run
            pass
        else:
            raise NotImplementedError(f'Unknown arg eval_during_train: {args.eval_during_train}')

    logging.info('Training done')

    # writer.close()

    return save_dir

def eval_during_train(args, save_dir, train_task_id, eval_results=None, writer=None):
    """
    Evaluates one saved model after training for 
    that task is complete.

    This avoids the need to save the networks for each task 
    for the purpose of evaluation.
    """

    # Check if cuda is available
    cuda_available, device = check_cuda()
    logging.info(f'cuda_available: {cuda_available}')

    # Dict for storing evaluation results
    # This will be written to a json file in the log folder
    # Create this if this is the first time eval is run
    if eval_results is None:
        eval_results = dict()

        # For storing command line arguments for this run
        eval_results['args'] = read_dict(os.path.join(save_dir, 'commandline_args.json'))

        # For storing the evaluation results
        eval_results['data'] = {'metrics': dict(), 'metric_errors': dict()}

    # Create a target network without parameters
    # Parameters are overwritten during the forward pass of the hypernetwork
    properties = {"latent_space_dim":args.tnet_dim,
                  "explicit_time": args.explicit_time,
                  "a":args.lsddm_a,
                  "projfn":args.lsddm_projfn,
                  "projfn_eps":args.lsddm_projfn_eps,
                  "smooth_v":args.lsddm_smooth_v,
                  "hp":args.lsddm_hp,
                  "h":args.lsddm_h,
                  "rehu":args.lsddm_rehu,
                  "device": device,
                  "fhat_layers": args.fhat_layers}

    # Create a LSDDM network 
    # Parameters are supplied during the forward pass of the hypernetwork
    # Load the network for the current task_id
    if args.explicit_time==1:
        target_network = configure_t(properties)
    elif args.explicit_time==0:
        target_network = configure(properties)    

    target_network = target_network.to(device)

    # Shapes of the target network parameters
    param_names, param_shapes = target_network.get_param_shapes()

    # Create the chunked hypernetwork
    if args.chn_type == 'complex':
        hnet = ChunkedHyperNetwork(final_target_shapes=param_shapes,
                                   layers=str_to_ints(args.hnet_arch),
                                   chunk_dim=args.chunk_dim,
                                   init_fn=str_to_init(args.init),
                                   te_dim=args.task_emb_dim,
                                   ce_dim=args.chunk_emb_dim,
                                   device=device).to(device)
    elif args.chn_type == 'simple':
        hnet = ChunkedTaskEmbHypernetwork(target_shapes=param_shapes, 
                                          layers=str_to_ints(args.hnet_arch), 
                                          te_dim=args.task_emb_dim,
                                          init_fn=str_to_init(args.init),
                                          dropout_rate=-1, 
                                          device=device,
                                          chunk_dim=args.chunk_dim)
    else:
        raise NotImplementedError(f'Unknown chn_type: {args.chn_type}')

    # The NODE uses the target network as the RHS of its
    # differential equation
    # Apart from this, the NODE has no other trainable parameters
    node = NODE(target_network=target_network, method='euler', explicit_time=args.explicit_time).to(device)

    # Extract the list of demonstrations from the text file 
    # containing the sequence of demonstrations
    seq = get_sequence(args.seq_file)

    num_tasks = len(seq)

    logging.info(f'#### Evaluation started for task_id: {train_task_id} (task {train_task_id+1} out of {num_tasks}) ###')

    eval_results['data']['metrics'][f'train_task_{train_task_id}'] = dict()
    eval_results['data']['metric_errors'][f'train_task_{train_task_id}'] = dict()

    # Load the network for the current task_id
    hnet = torch.load(os.path.join(save_dir, 'models', f'hnet_{train_task_id}.pth'))

    # Evaluate on all the past and current task_ids
    for eval_task_id in range(train_task_id+1):
        logging.info(f'Loaded network trained on task {train_task_id}, evaluating on task {eval_task_id}')

        
        eval_traj_metrics, eval_traj_metric_errors, node = eval_task(args, eval_task_id, hnet, target_network, node, device, param_names, train_task_id, writer)

        
        logging.info(f'Evaluated trajectory metrics: {eval_traj_metrics}')

        # Store the evaluated metrics
        eval_results['data']['metrics'][f'train_task_{train_task_id}'][f'eval_task_{eval_task_id}'] = eval_traj_metrics
        eval_results['data']['metric_errors'][f'train_task_{train_task_id}'][f'eval_task_{eval_task_id}'] = eval_traj_metric_errors

    # (Over)write the evaluation results to a file in the log dir
    write_dict(os.path.join(save_dir, 'eval_results.json'), eval_results)

    # Remove the networks that have been evaluated (except for the network of the last task)
    if train_task_id < (num_tasks-1):
        os.remove(os.path.join(save_dir, 'models', f'hnet_{train_task_id}.pth'))

    logging.info('Current task evaluation done')

    return eval_results

if __name__ == '__main__':

    # Parse commandline arguments
    args = parse_args()

    # Set the seed for reproducability
    set_seed(args.seed)

    # Training
    save_dir = train_all(args)

    # Evaluation can be run in a standalone manner if needed
    if args.eval_during_train == 1:
        raise NotImplementedError('eval_during_train=1 is not supported')
        #args = Dictobject(read_dict(os.path.join(save_dir, 'commandline_args.json')))
        #eval_all(args, save_dir)

    logging.info('Completed')