import numpy as np
from hypernet_explore.metrics.ori_metrics import convert_to_unit_quat, quat_distance

def pos_goal_stability(ref_trj_l, pred_trj_l, agg_type='mean'):
    """For two sets of position trajectories, this function computes the average distance 
    between the final points of the trajectories. Ideally, the prediction trajectories
    should be around 10% longer than the ground truth to understand if the prediction
    stays at or close to the goal.

    Args:
        ref_trj_l (np array): unnormalized ground truth with shape (num_traj, num_points, point_dim)
        pred_trj_l (np array): unnormalized prediction with shape (num_traj, num_points_pred, point_dim)

    Returns:
        np array: Mean distance between the goals of the ground truth and predicted trajectories
    """

    ref_shape = ref_trj_l.shape
    pred_shape = pred_trj_l.shape

    # Number of trajectories should match
    assert ref_shape[0]==pred_shape[0]

    # Data dimension should match
    assert ref_shape[2]==pred_shape[2]

    num_traj = ref_shape[0]

    # Final points of the trajectories
    gt_end = ref_trj_l[:,-1,:]
    pred_end = pred_trj_l[:,-1,:]

    # Euclidean distance between final points (1 value for each individual pair of trajectories)
    goal_diffs = np.linalg.norm((gt_end - pred_end), axis=-1)

    # Aggregated distance for all trajectories
    if agg_type == 'mean':
        goal_diff_agg = np.mean(goal_diffs)
    elif agg_type == 'median':
        goal_diff_agg = np.median(goal_diffs)

    return goal_diff_agg, goal_diffs

def ori_goal_stability(ref_trj_l, pred_trj_l, agg_type='mean'):
    # Distance to goal should be in terms of orientation error
    #TODO
    """For two sets of position trajectories, this function computes the average difference 
    between the final rotations of the trajectories. Ideally, the prediction trajectories
    should be around 10% longer than the ground truth to understand if the prediction
    stays at or close to the goal.

    Args:
        ref_trj_l (np array): unnormalized ground truth with shape (num_traj, num_points, point_dim)
        pred_trj_l (np array): unnormalized prediction with shape (num_traj, num_points_pred, point_dim)

    Returns:
        np array: Mean distance between the goals of the ground truth and predicted trajectories
    """

    ref_shape = ref_trj_l.shape
    pred_shape = pred_trj_l.shape

    # Number of trajectories should match
    assert ref_shape[0]==pred_shape[0]

    # Data dimension should match
    assert ref_shape[2]==pred_shape[2]

    num_demos = ref_shape[0]

    # Ensure unit quaternions
    pred_trj_l = convert_to_unit_quat(pred_trj_l)
    pred_trj_l = convert_to_unit_quat(pred_trj_l)

    # Final points of the trajectories
    gt_end = ref_trj_l[:,-1,:]
    pred_end = pred_trj_l[:,-1,:]

    # Rotation difference between final points (1 value for each individual pair of trajectories)
    ori_errors = list()
    for demo in range(num_demos):
        err = quat_distance(gt_end[demo], pred_end[demo])
        # Mean error for the whole trajectory
        ori_errors.append(err)
    ori_errors = np.array(ori_errors)

    # Aggregated distance for all trajectories
    if agg_type == 'mean':
        goal_diff_agg = np.mean(ori_errors)
    elif agg_type == 'median':
        goal_diff_agg = np.median(ori_errors)

    return goal_diff_agg, ori_errors
