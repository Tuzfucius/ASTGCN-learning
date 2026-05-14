from astgcn.data.io import load_pems_npz

data = load_pems_npz("data/raw/PEMS04/pems04.npz")

print("data shape:", data.shape)
print("time steps:", data.shape[0])
print("num nodes:", data.shape[1])
print("num features:", data.shape[2])