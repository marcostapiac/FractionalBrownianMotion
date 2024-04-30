from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyarrow.parquet import ParquetFile
import pyarrow as pa
config = get_config()
# Now plot Hurst histogram for the generated samples
train_epoch = 2920
score_error_path = (config.experiment_path.replace("results/",
                                                   "results/score_errors/") + "_NEp{}".format(
    train_epoch).replace(
    ".", "") + ".parquet.gzip").replace("rec_TSM_False_incs_True_unitIntv_", "")
pf = ParquetFile(score_error_path.replace(".parquet.gzip", "_bad1.parquet.gzip"))
first_ten_rows = next(pf.iter_batches(batch_size = config.max_diff_steps*10))
bad_score_df_1 = pa.Table.from_batches([first_ten_rows]).to_pandas()


bad_score_1= bad_score_df_1.to_numpy().reshape((bad_score_df_1.index.levshape[0], bad_score_df_1.index.levshape[1], bad_score_df_1.shape[1]))
avg_bad_score_1  = np.mean(bad_score_1,axis=-1)

timeav_bad_score_1 = np.mean(avg_bad_score_1,axis=0)
plt.plot(np.linspace(0, 20, 20), timeav_bad_score_1[:20])
plt.show()

plt.plot(np.linspace(5000, config.max_diff_steps, config.max_diff_steps-5000), timeav_bad_score_1[5000:])
plt.show()

plt.plot(np.linspace(8000, config.max_diff_steps, config.max_diff_steps-8000), timeav_bad_score_1[8000:])
plt.show()