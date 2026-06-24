import mne
import numpy as np

subjects = [1,2,3,4,5]
runs = [4,8,12]
X_list, y_list = [], []
for subject in subjects:
    print(f"Downloading and Processing Subject {subject}...")
    raw_fnames = mne.datasets.eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
    
    for raw in raws:
        mne.datasets.eegbci.standardize(raw)
        
    raw = mne.concatenate_raws(raws)
    events, _ = mne.events_from_annotations(raw)
    
    epochs = mne.Epochs(raw, events, tmin=0, tmax=5.0, baseline=None, preload=True)
    
    X = epochs.get_data()
    y = epochs.events[:, -1]
    
    # Mask for Left Fist (2) and Right Fist (3)
    mask = (y == 2) | (y == 3)
    X_list.append(X[mask])
    y_list.append(y[mask])

X_massive = np.concatenate(X_list, axis=0)
y_massive = np.concatenate(y_list, axis=0)

# Save the scaled data to the SSD
np.save("X_massive.npy", X_massive)
np.save("y_massive.npy", y_massive)

print(f"\nMassive X Tensor Shape: {X_massive.shape}")
