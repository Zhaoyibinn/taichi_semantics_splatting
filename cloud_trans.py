import pandas as pd
import numpy as np

cloud = pd.read_parquet("logs/tat_train_experiment_downsample_warmup/best_scene.parquet")
np.savetxt("logs/tat_train_experiment_downsample_warmup/cloud.txt",np.array(cloud))


print(cloud)