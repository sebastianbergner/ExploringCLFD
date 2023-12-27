import torch
import pandas as pd

def get_minibatch(t, y, nsub=None, tsub=None, dtype=torch.float64):
    """
    Extract nsub sequences each of lenth tsub from the original dataset y.

    Args:
        t (np array [T]): T integration time points from the original dataset.
        y (np array [N,T,d]): N observed sequences from the original dataset, 
                              each with T datapoints where d is the dimension 
                              of each datapoint.
        nsub (int): Number of sequences to be selected from.
                    If Nsub is None, then all N sequences are considered.
        tsub (int): Length of sequences to be returned.
                    If tsub is None, then sequences of length T are returned.

    Returns:
        tsub (torch tensor [tsub]): Integration time points (in this minibatch)
        ysub (torch tensor [nsub, tsub, d]): Observed (sub)sequences (in this minibatch)
    """
    # Find the number of sequences N and the length of each sequence T
    [N,T] = y.shape[:2]

    # If nsub is None, then consider all sequences
    # Else select nsub sequences randomly
    y_   = y if nsub is None else y[torch.randperm(N)[:nsub]]

    # Choose the starting point of the sequences
    # If tsub is None, then start from the beginning
    # Else find a random starting point based on tsub
    t0   = 0 if tsub is None else torch.randint(0,1+len(t)-tsub,[1]).item()  # pick the initial value
    tsub = T if tsub is None else tsub

    # Set the data to be returned
    tsub, ysub = torch.from_numpy(t[t0:t0+tsub]).type(dtype), torch.from_numpy(y_[:,t0:t0+tsub]).type(dtype)
    return tsub, ysub 

def get_minibatch_extended(t, y, nsub=None, tsub=None, dtype=torch.float64):
    """
    Extract nsub sequences each of lenth tsub from the original dataset y.

    Args:
        t (np array [T]): T integration time points from the original dataset.
        y (np array [N,T,d]): N observed sequences from the original dataset, 
                              each with T datapoints where d is the dimension 
                              of each datapoint.
        nsub (int): Number of sequences to be selected from.
                    If Nsub is None, then all N sequences are considered.
        tsub (int): Length of sequences to be returned.
                    If tsub is None, then sequences of length T are returned.

    Returns:
        tsub (torch tensor [tsub]): Integration time points (in this minibatch)
        ysub (torch tensor [nsub, tsub, d]): Observed (sub)sequences (in this minibatch)
    """
    # Find the number of sequences N and the length of each sequence T
    [N,T] = y.shape[:2]

    # If nsub is None, then consider all sequences
    # Else select nsub sequences randomly
    y_   = y if nsub is None else y[torch.randperm(N)[:nsub]]

    # Choose the starting point of the sequences
    # If tsub is None, then start from the beginning
    # Else find a random starting point based on tsub
    t0   = 0 if tsub is None else torch.randint(0,1+len(t)-tsub,[1]).item()  # pick the initial value
    tsub = T if tsub is None else tsub

    # Set the data to be returned
    tsub, ysub = t[t0:t0+tsub], y_[:,t0:t0+tsub]

    if not torch.is_tensor(tsub):
        tsub = torch.from_numpy(tsub)

    if not torch.is_tensor(ysub):
        ysub = torch.from_numpy(ysub)

    return tsub, ysub 

def get_dtw_summary(dataset, data_dim, norm, datafile='datasets/ground_truth_dtw.csv', verbose=False):
    """
    Used for plotting the DTW thresholds from ground truth (for comparison)
    A cumulative median of the mean gt dtw value of each task is returned
    """

    # Load dtw information
    all_datasets_df = pd.read_csv(datafile)
    
    dtw_summary = all_datasets_df.groupby(["dataset", "data_dim", "norm", "task_id"]).agg(
                                            max_dtw=pd.NamedAgg(column='dtw', aggfunc='max'),
                                            min_dtw=pd.NamedAgg(column='dtw', aggfunc='min'),
                                            mean_dtw=pd.NamedAgg(column='dtw', aggfunc='mean'),
                                            std_dtw=pd.NamedAgg(column='dtw', aggfunc='std'),
                                            median_dtw=pd.NamedAgg(column='dtw', aggfunc='median')
                                            ).round(2).query(f"(dataset=='{dataset}') and (data_dim=={data_dim}) and (norm=={norm})")
    # Add rolling median for max_dtw
    if data_dim > 2:
        dtw_summary['cumu_thresh_dtw'] = (dtw_summary['mean_dtw']).expanding().median()
    else:
        dtw_summary['cumu_thresh_dtw'] = (dtw_summary['mean_dtw']).expanding().median()

    if verbose:
        pd.set_option('display.max_rows', None)
        print(dtw_summary)

    return dtw_summary['cumu_thresh_dtw'].to_list()