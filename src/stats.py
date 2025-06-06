import pandas as pd

df = pd.read_csv('stats.txt', sep=', ')
print(df)
print(df.keys())
## for each GPUidx calcualte the mean and std of the Mean power usage and the mean and std of the MAX power usage
for pdb, pdb_df in df.groupby('PDB'):
    #print(f"PDB: {pdb}")
    #print(pdb_df)
    # for each GPUidx calculate the mean and std of the Mean power usage and the mean and std of the MAX power usage
    for gpu_idx, gpu_idx_df in pdb_df.groupby('GPUidx'):
        mean_power = gpu_idx_df['Mean'].mean()
        std_power = gpu_idx_df['Mean'].std()
        max_power = gpu_idx_df['Max'].mean()
        std_max_power = gpu_idx_df['Max'].std()
        print(f"PDB: {pdb} GPU {gpu_idx}: Mean Power: {mean_power:.2f} W, Std: {std_power:.2f} W, Max Power: {max_power:.2f} W, Std: {std_max_power:.2f} W")