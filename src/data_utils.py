from scipy.signal import resample_poly
from src.config import DOWNSAMPLE_DOWN, DOWNSAMPLE_UP, WINDOWS

def downsample_eeg(X): 
    X_250 = resample_poly(
        X, 
        up=DOWNSAMPLE_UP, 
        down=DOWNSAMPLE_DOWN, 
        axis=-1, 
        padtype='line'
    )
    return X_250.astype('float32')

def select_window(X, window_name, sfreq:int, windows=WINDOWS): 
    return X[:, :, windows[sfreq][window_name]]