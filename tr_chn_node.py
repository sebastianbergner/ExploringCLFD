import os
from tqdm import trange
from argparse import ArgumentParser
import logging
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# from tensorboardX import SummaryWriter

from hypernet_explore.train.utils import check_cuda, set_seed, get_sequence, get_beta_for_tasks
from hypernet_explore.model.hypernetwork import ChunkedHyperNetwork, ChunkedTaskEmbHypernetwork, TargetNetwork, \
    str_to_ints, str_to_act, get_current_targets, calc_delta_theta, calc_fix_target_reg, str_to_init, str_to_optim
from hypernet_explore.model.node import NODE
from hypernet_explore.data.lasa import LASA, LASAExtended
from hypernet_explore.data.helloworld import HelloWorld
from hypernet_explore.data.robottasks import RobotTasksPosition, RobotTasksOrientation, RobotTasksPositionOrientation
from hypernet_explore.data.utils import get_minibatch, get_minibatch_extended
from hypernet_explore.metrics.traj_metrics import mean_swept_error, mean_frechet_error_fast as mean_frechet_error, dtw_distance_fast as dtw_distance
from hypernet_explore.metrics.ori_metrics import quat_traj_distance
from hypernet_explore.logging.utils import custom_logging_setup, write_dict, read_dict, Dictobject
from hypernet_explore.metrics.entropy import batch_entropy

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
    parser.add_argument('--tnet_arch', type=str, default='200,200,200', help='Hidden layer units of the target network')
    parser.add_argument('--tnet_act', type=str, default='elu', help='Target network activation function')
    parser.add_argument('--init', type=str, default='kaiming', help='Initialization function')
    parser.add_argument('--init_bias_pwi', type=int, default=0, help='Use PWI bias initialization (only used with PWI)')
    parser.add_argument('--hnet_arch', type=str, default='200,200,200', help='Hidden layer units of the hypernetwork')
    parser.add_argument('--task_emb_dim', type=int, default=5, help='Dimension of the task embedding vector')
    parser.add_argument('--chunk_emb_dim', type=int, default=5, help='Dimension of the each chunk embedding vector (input to the HN)')
    parser.add_argument('--chunk_dim', type=int, default=1000, help='Dimension of the output of the chunked HN (these chunks are tiled together to create the final target network)')
    parser.add_argument('--explicit_time', type=int, default=0, help='1: Use time as an explicit network input, 1: Do not use time')
    parser.add_argument('--chn_type', type=str, default='complex', help='complex: CHN with a separate regularized chunk embedding, complex: CHN with a chunked task embedding and no separate chunk embedding')
    parser.add_argument('--beta_decay', type=float, default=1.0, help='Multiplicative factor for beta (for each task: beta *= beta_decay')

    parser.add_argument('--zero_center', type=int, default=0, help='0: Do not center trajectories, 1: Center trajectories before learning')

    parser.add_argument('--int_method', type=str, default='euler', help='Integration method')

    # Additional hyperparameters
    parser.add_argument('--optimizer', type=str, default='Adam', help='Type of optimizer to use (Adam, RMSProp or SGD)')
    parser.add_argument('--dropout', type=float, default=-1.0, help='Dropout rate. -1.0 means no dropout')

    parser.add_argument('--beta', type=float, default=5e-3, help='Regularization strength')

    parser.add_argument('--lr_change_iter', type=int, default=-1, help='-1 or 0: No LR scheduler, >0: Number of iterations after which initial LR is divided by 10')

    parser.add_argument('--data_class', type=str, required=True, help='Dataset class for training')
    parser.add_argument('--eval_during_train', type=int, default=0, help='0: net for a task is evaluated immediately after training, 1: eval for all nets is done after training of all tasks')
    parser.add_argument('--seed', type=int, required=True, help='Seed for reproducability')
    parser.add_argument('--seq_file', type=str, required=True, help='Name of file containing sequence of demonstration files')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Main directory for saving logs')
    parser.add_argument('--description', type=str, required=True, help='String identifier for experiment')

    # Old (data in mat files) or new (data in numpy archives) data loading process
    parser.add_argument('--data_type', type=str, default='mat', help='Type of data to load from - mat: mat files, np: numpy archives')

    # Scaling term for tangent vectors for learning orientation
    parser.add_argument('--tangent_vec_scale', type=float, default=1.0, help='Tangent vector scaling term')

    if return_parser:
        # This is used by the slurm creator script
        # When running this script directly, this has no effect
        return parser
    else:
        args = parser.parse_args()
        return args

def train_task(args, task_id, hnet, tnet, node, beta, device, writer=None, save_dir=""):

    filenames = get_sequence(args.seq_file)

    assert 0.0<=beta<=1.0, f'Invalid beta: {beta}'

    data = None
    if args.data_class == 'LASA':
        if args.data_type == 'mat':
            data = LASA(data_dir=args.data_dir, filename=filenames[task_id], replicate_num=args.replicate_num)
        elif args.data_type == 'np':
            datafile = os.path.join(args.data_dir, filenames[task_id])
            data = LASAExtended(datafile, seq_len=args.tsub, norm=True, device=device)

            if args.zero_center == 1:
                # Goal position at origin
                data.zero_center()
        else:
            raise NotImplementedError(f'data_type {args.data_type} not available for data_class {args.data_class}')
    elif args.data_class == 'HelloWorld':
        data = HelloWorld(data_dir=args.data_dir, filename=filenames[task_id])
        if args.zero_center == 1:
            # Goal position at origin
            data.zero_center()
    elif args.data_class == 'RobotTasksPosition':
        if args.data_type == 'np':
            data = RobotTasksPosition(data_dir=args.data_dir, datafile=filenames[task_id], device=device)
            if args.zero_center == 1:
                # Goal position at origin
                data.zero_center()
        else:
            raise NotImplementedError(f'data_type {args.data_type} not available for data_class {args.data_class}')
    elif args.data_class == 'RobotTasksOrientation':
        if args.data_type == 'np':
            data = RobotTasksOrientation(data_dir=args.data_dir, datafile=filenames[task_id], device=device, scale=args.tangent_vec_scale)
        else:
            raise NotImplementedError(f'data_type {args.data_type} not available for data_class {args.data_class}')
    elif args.data_class == 'RobotTasksPositionOrientation':
        if args.data_type == 'np':
            data = RobotTasksPositionOrientation(data_dir=args.data_dir, datafile=filenames[task_id], device=device, scale=args.tangent_vec_scale)
            if args.zero_center == 1:
                # Goal position at origin
                data.zero_center()
        else:
            raise NotImplementedError(f'data_type {args.data_type} not available for data_class {args.data_class}')
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
    optim_fn = str_to_optim(args.optimizer)
    theta_optimizer = optim_fn(regularized_params, lr=args.lr)

    # Apply learning scheduler if needed
    if args.lr_change_iter > 0:
        theta_lambda = lambda epoch: 1.0 if (epoch < args.lr_change_iter) else 0.1
        theta_scheduler = LambdaLR(theta_optimizer, lr_lambda=theta_lambda)
    
    # For optimizing the task embedding for the current task.
    # We only optimize the task embedding corresponding to the current task,
    # the remaining ones stay constant.
    
    emb_optimizer = optim_fn([hnet.get_task_emb(task_id)], lr=args.lr)

    # Whether the regularizer will be computed during training?
    calc_reg = task_id > 0 and args.beta > 0

    best_loss = np.inf
    best_iter = 0
    #best_hnet = deepcopy(hnet) # In case of nans its better if an error is thrown

    hnet.init_batch_entropy_new_task()

    # Start training iterations
    for training_iters in trange(args.num_iter):

        # Set flag for storing batch entropy
        hnet.compute_batch_entropy = True

        ### Train theta and task embedding
        theta_optimizer.zero_grad()
        emb_optimizer.zero_grad()

        # Generate parameters of the target network for the current task
        weights = hnet.forward(task_id)

        # Set the weights of the target network
        tnet.set_weights(weights)

        # Set the target network in the NODE
        node.set_target_network(tnet)

        if args.data_type == 'mat':
            t, y_all = get_minibatch(data.t[0], data.pos, tsub=args.tsub)
        elif args.data_type == 'np':
            t, y_all = get_minibatch_extended(data.t[0], data.pos, nsub=None, tsub=args.tsub, dtype=torch.float)

        # The time steps
        t = t.to(device)

        # Subsequence trajectories
        y_all = y_all.to(device)

        # Starting points
        y_start = y_all[:,0].float()

        # Predicted trajectories - forward simulation
        y_hat = node(t.float(), y_start) 
        
        # MSE
        loss = ((y_hat-y_all)**2).mean()

        # Log the loss in tensorboard
        if writer is not None:
            writer.add_scalar(f'task_loss/task_{task_id}', loss.item(), training_iters)

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
                writer.add_scalar(f'reg_loss/task_{task_id}', loss_reg.item(), training_iters)

            # Backpropagate the regularization loss
            loss_reg.backward()

        # Update the hnet params using the current task loss and the regularization loss
        theta_optimizer.step()

        if args.lr_change_iter > 0:
            theta_scheduler.step()
        
        if loss.item() <= best_loss:
            best_hnet = deepcopy(hnet)
            best_loss = loss.item()
            best_iter = int(training_iters)

    hnet.store_batch_entropy(save_dir, task_id)
       
    return best_hnet, best_loss, best_iter

def eval_task(args, task_id, hnet, tnet, node, device, writer=None):

    hnet.eval()
    tnet.eval()

    tnet = tnet.to(device)
    hnet = hnet.to(device)
    node = node.to(device)

    filenames = get_sequence(args.seq_file)

    data = None
    if args.data_class == 'LASA':
        if args.data_type == 'mat':
            data = LASA(data_dir=args.data_dir, filename=filenames[task_id], replicate_num=args.replicate_num)
        elif args.data_type == 'np':
            datafile = os.path.join(args.data_dir, filenames[task_id])
            data = LASAExtended(datafile, seq_len=args.tsub, norm=True, device=device)
            if args.zero_center == 1:
                # Goal position at origin
                data.zero_center()
        else:
            raise NotImplementedError(f'data_type {args.data_type} not available for data_class {args.data_class}')
    elif args.data_class == 'HelloWorld':
        data = HelloWorld(data_dir=args.data_dir, filename=filenames[task_id])
        if args.zero_center == 1:
            # Goal position at origin
            data.zero_center()
    elif args.data_class == 'RobotTasksPosition':
        if args.data_type == 'np':
            data = RobotTasksPosition(data_dir=args.data_dir, datafile=filenames[task_id], device=device)
            if args.zero_center == 1:
                # Goal position at origin
                data.zero_center()        
        else:
            raise NotImplementedError(f'data_type {args.data_type} not available for data_class {args.data_class}')
    elif args.data_class == 'RobotTasksOrientation':
        if args.data_type == 'np':
            data = RobotTasksOrientation(data_dir=args.data_dir, datafile=filenames[task_id], device=device, scale=args.tangent_vec_scale)
        else:
            raise NotImplementedError(f'data_type {args.data_type} not available for data_class {args.data_class}')
    elif args.data_class == 'RobotTasksPositionOrientation':
        if args.data_type == 'np':
            data = RobotTasksPositionOrientation(data_dir=args.data_dir, datafile=filenames[task_id], device=device, scale=args.tangent_vec_scale)
            if args.zero_center == 1:
                # Goal position at origin
                data.zero_center()
        else:
            raise NotImplementedError(f'data_type {args.data_type} not available for data_class {args.data_class}')
    else:
        raise NotImplementedError(f'Unknown dataset class {args.data_class}')

    # Generate parameters of the target network for the current task
    weights = hnet.forward(task_id)

    # Set the weights of the target network
    tnet.set_weights(weights)

    # Set the target network in the NODE
    node.set_target_network(tnet)
    node = node.float()
    node.eval()

    if args.data_type == 'mat':
        # The time steps
        t = torch.from_numpy(data.t[0]).float().to(device)

        # The starting position 
        # (n,d-dimensional, where n is the num of demos and 
        # d is the dimension of each point)
        y_start = torch.from_numpy(data.pos[:,0]).float().to(device)

        # The entire demonstration trajectory
        y_all = torch.from_numpy(data.pos).float().to(device)
    elif args.data_type == 'np':
        # The time steps
        t = data.t[0].float()

        # The starting position 
        # (n,d-dimensional, where n is the num of demos and 
        # d is the dimension of each point)
        #y_start = torch.unsqueeze(dataset.pos[0,0], dim=0)
        # Use the translated trajectory (goal at origin)
        y_start = data.pos[:,0]
        y_start = y_start.float()

        # The entire demonstration trajectory
        y_all = data.pos.float()

    # Predicted trajectory
    if args.zero_center == 1:
        y_hat_zeroed = node(t, y_start)
        y_hat = data.unzero_center(y_hat_zeroed)
    else:
        y_hat = node(t, y_start) # forward simulation

    # Compute trajectory metrics
    y_all_np = y_all.cpu().detach().numpy()
    y_hat_np = y_hat.cpu().detach().numpy()

    # De-normalize the data before computing trajectories
    y_all_np = data.unnormalize(y_all_np)
    y_hat_np = data.unnormalize(y_hat_np)    

    if args.data_class == 'RobotTasksOrientation':
        # Convert predicted trajectory from tangent vectors to quaternions
        q_hat_np = data.from_tangent_plane(y_hat_np)

        # Compare the predicted quaternion trajectorywith the ground truth
        metric_quat_err, metric_quat_errs = quat_traj_distance(data.quat_data, q_hat_np)

        eval_traj_metrics = {'quat_error': metric_quat_err}
        eval_traj_metric_errors = {'quat_error': metric_quat_errs.tolist()}
    elif args.data_class == 'RobotTasksPositionOrientation':
        # Separate the position and rotation vectors
        # Predictions
        position_hat_np = y_hat_np[:,:,:3]
        rotation_hat_np = y_hat_np[:,:,3:]
        # Ground truth
        position_all_np = y_all_np[:,:,:3]
        rotation_all_np = y_all_np[:,:,3:]

        # Convert predicted rotation trajectory from tangent vectors to quaternions
        q_hat_np = data.from_tangent_plane(rotation_hat_np)

        # Compute metrics for position
        metric_swept_err, metric_swept_errs = mean_swept_error(position_all_np, position_hat_np)
        metric_frechet_err, metric_frechet_errs = mean_frechet_error(position_all_np, position_hat_np)
        metric_dtw_err, metric_dtw_errs = dtw_distance(position_all_np, position_hat_np)

        # Compute metrics for quaternion
        metric_quat_err, metric_quat_errs = quat_traj_distance(data.rotation_quat, q_hat_np)

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
        # Calculate trajectory metrics
        metric_swept_err, metric_swept_errs = mean_swept_error(y_all_np, y_hat_np)
        metric_frechet_err, metric_frechet_errs = mean_frechet_error(y_all_np, y_hat_np)
        metric_dtw_err, metric_dtw_errs = dtw_distance(y_all_np, y_hat_np)

        eval_traj_metrics = {'swept': metric_swept_err, 
                            'frechet': metric_frechet_err, 
                            'dtw': metric_dtw_err}

        # Store the metric errors
        # Convert np arrays to list so that these can be written to JSON
        eval_traj_metric_errors = {'swept': metric_swept_errs.tolist(), 
                                'frechet': metric_frechet_errs.tolist(), 
                                'dtw': metric_dtw_errs.tolist()}

    return eval_traj_metrics, eval_traj_metric_errors


def train_all(args):

    # Create logging folder and set up console logging
    save_dir, identifier = custom_logging_setup(args)

    # Tensorboard logging setup
    # writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tb', args.description, identifier))

    # Check if cuda is available
    cuda_available, device = check_cuda()
    logging.info(f'cuda_available: {cuda_available}')
        

    # Shapes of the target network parameters
    target_shapes = TargetNetwork.weight_shapes(n_in=args.tnet_dim+args.explicit_time, 
                                                n_out=args.tnet_dim, 
                                                hidden_layers=str_to_ints(args.tnet_arch), 
                                                use_bias=True)

    # Create the chunked hypernetwork
    if args.chn_type == 'complex':
        hnet = ChunkedHyperNetwork(final_target_shapes=target_shapes,
                                   layers=str_to_ints(args.hnet_arch),
                                   chunk_dim=args.chunk_dim,
                                   init_fn=str_to_init(args.init),
                                   init_bias_pwi=args.init_bias_pwi,
                                   te_dim=args.task_emb_dim,
                                   ce_dim=args.chunk_emb_dim,
                                   dropout_rate=args.dropout,
                                   verbose=True,
                                   device=device).to(device)
    elif args.chn_type == 'simple':
        hnet = ChunkedTaskEmbHypernetwork(target_shapes=target_shapes, 
                                          layers=str_to_ints(args.hnet_arch), 
                                          te_dim=args.task_emb_dim,
                                          init_fn=str_to_init(args.init),
                                          init_bias_pwi=args.init_bias_pwi,
                                          dropout_rate=-1, 
                                          device=device,
                                          chunk_dim=args.chunk_dim)
    else:
        raise NotImplementedError(f'Unknown chn_type: {args.chn_type}')
        
    # Create a target network without parameters
    # Parameters are supplied during the forward pass of the hypernetwork
    tnet = TargetNetwork(n_in=args.tnet_dim+args.explicit_time, 
                         n_out=args.tnet_dim, 
                         hidden_layers=str_to_ints(args.tnet_arch),
                         activation_fn=str_to_act(args.tnet_act),
                         init_fn=str_to_init(args.init),
                         init_bias_pwi=args.init_bias_pwi,
                         use_bias=True, 
                         no_weights=True,
                         init_weights=None, 
                         dropout_rate=-1,  # Dropout is only used for hnet 
                         use_batch_norm=False, 
                         bn_track_stats=False,
                         distill_bn_stats=False, 
                         out_fn=None,
                         device=device).to(device)

    # The NODE uses the target network as the RHS of its
    # differential equation
    # Apart from this, the NODE has no other trainable parameters
    node = NODE(tnet, explicit_time=args.explicit_time, method=args.int_method).to(device)

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
        hnet, best_loss, best_iter = train_task(args, task_id, hnet, tnet, node, betas[task_id], device, writer=None, save_dir=save_dir)

        logging.info(f'task_id: {task_id}, best_loss: {best_loss:.3E}, best_iter: {best_iter}')

        # At the end of every task store the latest networks
        logging.info('Saving models')
        torch.save(hnet, os.path.join(save_dir, 'models', f'hnet_{task_id}.pth'))

        if args.eval_during_train == 0:
            # Evaluate the latest network immediately after training
            # is complete for a task
            eval_resuts = eval_during_train(args, save_dir, task_id, eval_resuts)
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

    # Shapes of the target network parameters
    target_shapes = TargetNetwork.weight_shapes(n_in=args.tnet_dim+args.explicit_time, 
                                                n_out=args.tnet_dim, 
                                                hidden_layers=str_to_ints(args.tnet_arch), 
                                                use_bias=True)

    # Create the chunked hypernetwork
    if args.chn_type == 'complex':
        hnet = ChunkedHyperNetwork(final_target_shapes=target_shapes,
                                   layers=str_to_ints(args.hnet_arch),
                                   chunk_dim=args.chunk_dim,
                                   init_fn=str_to_init(args.init),
                                   te_dim=args.task_emb_dim,
                                   ce_dim=args.chunk_emb_dim,
                                   dropout_rate=args.dropout,
                                   device=device).to(device)
    elif args.chn_type == 'simple':
        hnet = ChunkedTaskEmbHypernetwork(target_shapes=target_shapes, 
                                          layers=str_to_ints(args.hnet_arch), 
                                          te_dim=args.task_emb_dim,
                                          init_fn=str_to_init(args.init),
                                          dropout_rate=-1, 
                                          device=device,
                                          chunk_dim=args.chunk_dim)
    else:
        raise NotImplementedError(f'Unknown chn_type: {args.chn_type}')
        
    # Create a target network without parameters
    # Parameters are supplied during the forward pass of the hypernetwork
    tnet = TargetNetwork(n_in=args.tnet_dim+args.explicit_time, 
                         n_out=args.tnet_dim, 
                         hidden_layers=str_to_ints(args.tnet_arch),
                         activation_fn=str_to_act(args.tnet_act),
                         init_fn=str_to_init(args.init),
                         use_bias=True, 
                         no_weights=True,
                         init_weights=None, 
                         dropout_rate=-1,  # Dropout is only used for hnet
                         use_batch_norm=False, 
                         bn_track_stats=False,
                         distill_bn_stats=False, 
                         out_fn=None,
                         device=device).to(device)

    # The NODE uses the target network as the RHS of its
    # differential equation
    # Apart from this, the NODE has no other trainable parameters
    node = NODE(tnet, explicit_time=args.explicit_time, method=args.int_method).to(device)

    # Extract the list of demonstrations from the text file 
    # containing the sequence of demonstrations
    seq = get_sequence(args.seq_file)

    num_tasks = len(seq)

    logging.info(f'#### Evaluation started for task_id: {train_task_id} (task {train_task_id+1} out of {num_tasks}) ###')

    eval_results['data']['metrics'][f'train_task_{train_task_id}'] = dict()
    eval_results['data']['metric_errors'][f'train_task_{train_task_id}'] = dict()

    # Load the networks for the current task_id
    hnet = torch.load(os.path.join(save_dir, 'models', f'hnet_{train_task_id}.pth'))
    #tnet = torch.load(os.path.join(save_dir, 'models', f'tnet_{train_task_id}.pth'))
    #node = torch.load(os.path.join(save_dir, 'models', f'node_{train_task_id}.pth'))

    # Evaluate on all the past and current task_ids
    for eval_task_id in range(train_task_id+1):
        logging.info(f'Loaded network trained on task {train_task_id}, evaluating on task {eval_task_id}')
        
        eval_traj_metrics, eval_traj_metric_errors = eval_task(args, eval_task_id, hnet, tnet, node, device, writer)
        
        logging.info(f'Evaluated trajectory metrics: {eval_traj_metrics}')

        # Store the evaluated metrics
        eval_results['data']['metrics'][f'train_task_{train_task_id}'][f'eval_task_{eval_task_id}'] = eval_traj_metrics
        eval_results['data']['metric_errors'][f'train_task_{train_task_id}'][f'eval_task_{eval_task_id}'] = eval_traj_metric_errors

    # (Over)write the evaluation results to a file in the log dir
    write_dict(os.path.join(save_dir, 'eval_results.json'), eval_results)

    # Remove the networks that have been evaluated (except for the network of the last task)
    if train_task_id < (num_tasks-1):
        os.remove(os.path.join(save_dir, 'models', f'hnet_{train_task_id}.pth'))
        #os.remove(os.path.join(save_dir, 'models', f'tnet_{train_task_id}.pth'))
        #os.remove(os.path.join(save_dir, 'models', f'node_{train_task_id}.pth'))

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
