#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from configs.RecursiveVPSDE.Markovian_8DLorenz.recursive_Markovian_PostMeanScore_8DLorenz_Stable_T256_H05_tl_110data_StbleTgt import get_config as get_8dlnz_config
from configs.RecursiveVPSDE.Markovian_12DLorenz.recursive_Markovian_PostMeanScore_12DLorenz_Stable_T256_H05_tl_110data_StbleTgt import get_config as get_12dlnz_config
from configs.RecursiveVPSDE.Markovian_20DLorenz.recursive_Markovian_PostMeanScore_20DLorenz_Stable_T256_H05_tl_110data_StbleTgt import get_config as get_20dlnz_config
from configs.RecursiveVPSDE.Markovian_40DLorenz.recursive_Markovian_PostMeanScore_40DLorenz_Stable_T256_H05_tl_110data_StbleTgt import get_config as get_40dlnz_config


# In[2]:


lnz_8d_config = get_8dlnz_config()
lnz_12d_config = get_12dlnz_config()
lnz_20d_config = get_20dlnz_config()
lnz_40d_config = get_40dlnz_config()
root_dir ="/Users/marcos/Library/CloudStorage/OneDrive-ImperialCollegeLondon/StatML_CDT/Year2/DiffusionModels/"


# In[3]:
import io

def get_best_epoch(type):
    model_dir = "/".join(config.scoreNet_trained_path.split("/")[:-1]) + "/"
    for file in os.listdir(model_dir):
        if config.scoreNet_trained_path in os.path.join(model_dir, file) and f"{type}" in file:
            print(file.split(f"{type}NEp")[-1])
            best_epoch = int(file.split(f"{type}NEp")[-1])
    return best_epoch
def get_best_eval_exp_file(root_score_dir, ts_type):
    best_epoch_eval = get_best_epoch(type="EE")
    for file in os.listdir(root_score_dir):
        if ("_"+str(best_epoch_eval)+"Nep") in file and "MSE" in file and ts_type in file and "1000FTh" in file and "075FConst" in file:
            print(f"Starting {file}\n")
            with open(root_score_dir+file, 'rb') as f:
                buf = io.BytesIO(f.read())  # hydrates once, sequentially
            print(f"Starting {file}\n")
            mse = pd.read_parquet(root_score_dir+file, engine="fastparquet")
    return mse

def get_best_track_file(root_score_dir, ts_type, best_epoch_track):
    for file in os.listdir(root_score_dir):
        if ("_"+str(best_epoch_track)+"Nep") in file and "true" in file and ts_type in file and "1000FTh" in file and "075FConst" in file:
            print(f"Starting {file}\n")
            with open(root_score_dir+file, 'rb') as f:
                buf = io.BytesIO(f.read())  # hydrates once, sequentially
            print(f"Starting {file}\n")
            true_file = np.load(root_score_dir+file, allow_pickle=True)
        elif ("_"+str(best_epoch_track)+"Nep") in file and "global" in file and ts_type in file and "1000FTh" in file and "075FConst" in file:
            print(f"Starting {file}\n")
            with open(root_score_dir+file, 'rb') as f:
                buf = io.BytesIO(f.read())  # hydrates once, sequentially
            print(f"Starting {file}\n")
            global_file = np.load(root_score_dir+file, allow_pickle=True)
    print(ts_type)
    return true_file, global_file

def track_pipeline(root_score_dir, ts_type, config, root_dir, toSave, label):
    best_epoch_track = get_best_epoch(type="Trk")
    all_true_states, all_global_states = get_best_track_file(root_score_dir=root_score_dir, ts_type=ts_type, best_epoch_track=best_epoch_track)
    print(all_true_states.shape)
    time_steps = np.linspace(config.t0,config.deltaT*all_true_states.shape[2],all_true_states.shape[2])
    all_global_errors = np.sum(np.power(all_true_states- all_global_states,2), axis=-1)
    all_global_errors = all_global_errors.reshape(-1, all_global_errors.shape[-1])
    total_global_errors = np.sqrt(np.mean((all_global_errors), axis=0))/np.sqrt(time_steps)
    all_errs = np.sqrt(all_global_errors)/np.sqrt(time_steps)
    total_global_errors[np.isinf(total_global_errors)] = 0.
    all_errs[np.isinf(all_errs)] = 0.
    total_global_errors_minq, total_global_errors_maxq = np.quantile(all_errs, axis=0,q=[0.005,0.995])
    fig, ax = plt.subplots(figsize=(14,9))
    plt.grid(True)
    ax.scatter(time_steps, total_global_errors)
    plt.fill_between(time_steps,y1=total_global_errors_minq, y2=total_global_errors_maxq, color="blue", alpha=0.4)
    ax.set_title(f"Pathwise RMSE for Score Estimator for {label}",fontsize=40)

    ax.set_xlabel("Time Axis", fontsize=38)
    ax.tick_params(labelsize=38)

    fig.canvas.draw()
    # Get the offset text (e.g., '1e-5')
    offset_text = ax.yaxis.get_offset_text().get_text()

    # Remove the offset text from the axis
    ax.yaxis.get_offset_text().set_visible(False)

    # Inject the scale into the y-axis label
    if offset_text:
        ax.set_ylabel(f'RMSE ({offset_text})', fontsize=38)
    else:
        ax.set_ylabel('RMSE', fontsize=38)
    plt.tight_layout()
    if toSave:
        plt.savefig((root_dir +f"DiffusionModelPresentationImages/TSPM_Markovian/{ts_type}LessData/TSPM_MLP_PM_ST_{config.feat_thresh:.3f}FTh_{ts_type}_DriftTrack_{best_epoch_track}Nep_{round(total_global_errors_minq[-1], 7)}_MinIQR_{round(total_global_errors_maxq[-1], 7)}_MaxIQR").replace(".", "")+".png")
    plt.grid(True)
    plt.show()
    plt.close()
    print(f"Final time cumulative MSE local-time error {total_global_errors[-1]} with IQR ({total_global_errors_minq[-1], total_global_errors_maxq[-1]})at Nepoch {best_epoch_track}\n")
    return total_global_errors[-1]


# In[ ]:


toSave = False
eval_tracks = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}
for config in [lnz_20d_config, lnz_40d_config]:
    assert config.feat_thresh == 1.
    assert config.forcing_const == 0.75
    Xshape = config.ts_length
    root_score_dir = root_dir
    label = "$\mu_{5}$"
    if "8DLnz" in config.data_path:
        root_score_dir = root_dir + f"ExperimentResults/TSPM_Markovian/8DLnzLessData/"
        ts_type = "8DLnz"
    elif "12DLnz" in config.data_path:
        root_score_dir = root_dir + f"ExperimentResults/TSPM_Markovian/12DLnzLessData/"
        ts_type = "12DLnz"
    elif "20DLnz" in config.data_path:
        root_score_dir = root_dir + f"ExperimentResults/TSPM_Markovian/20DLnzLessData/"
        ts_type = "20DLnz"
    elif "40DLnz" in config.data_path:
        root_score_dir = root_dir + f"ExperimentResults/TSPM_Markovian/40DLnzLessData/"
        ts_type = "40DLnz"
    print(f"Starting {ts_type}\n")
    rmse = get_best_eval_exp_file(root_score_dir=root_score_dir, ts_type=ts_type)
    eval_tracks[ts_type] = [rmse.values[0][0]]
    #rmse = track_pipeline(root_score_dir=root_score_dir, ts_type=ts_type, config=config, root_dir=root_dir, toSave=toSave, label=label)
    #eval_tracks[ts_type] = [rmse]


# In[ ]:


eval_tracks = (pd.DataFrame.from_dict(eval_tracks))
print(eval_tracks)


# In[ ]:




