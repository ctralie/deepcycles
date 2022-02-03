import cv2
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt


def load_video(filename, make_rgb=False, ycrcb=False, verbose=False, pow2=False):
    """
    Wraps around OpenCV to load a video frame by frame
    
    Parameters
    ----------
    filename: string
        Path to the video file I want to load
    make_rgb: bool
        If True, frames are in RGB order
        If False (default), frames are in BGR order
    ycrcb: bool
        If True, convert to ycrcb
    verbose: bool
        If True, print a . for every frame as it's loaded
    pow2: bool
        If True, zeropad video to nearest power of 2
    
    Returns
    -------
    ndarray(nrows, ncols, 3, nframes)
        A 4D array for the color video
    
    """
    cap = cv2.VideoCapture(filename)
    if (cap.isOpened()== False):
        print("Error opening file " + filename)
        return
    frames = []
    ret = True
    while ret and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if make_rgb:
                # Change to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif ycrcb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            frames.append(frame)
            if verbose:
                print('.', end='')
    frames = np.array(frames)
    # Put dimensions as 0:row, 1:col, 2: color, 3: frame num
    frames = np.moveaxis(frames, (0, 1, 2, 3), (3, 0, 1, 2))
    if verbose:
        print("")
    cap.release()
    if pow2:
        n_rows = int(2**np.ceil(np.log2(frames.shape[0])))
        n_cols = int(2**np.ceil(np.log2(frames.shape[1])))
        frames2 = np.zeros((n_rows, n_cols, frames.shape[2], frames.shape[3]), dtype=frames.dtype)
        dr = int((n_rows - frames.shape[0])/2)
        dc = int((n_cols - frames.shape[1])/2)
        frames2[dr:frames.shape[0]+dr, dc:frames.shape[1]+dc, :, :] = frames
        frames = frames2
    return frames

def save_video(filename, frames, fps=30, is_rgb=False, is_ycrcb=False):
    """
    Wraps around OpenCV to save a sequence of frames to a video file
    
    Parameters
    ----------
    filename: string
        Path to which to write video
    frames: ndarray(nrows, ncols, 3, nframes)
        A 4D array for the color video
    fps: int
        Frames per second of output video (default 30)
    """
    result = cv2.VideoWriter(filename, 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             fps, (frames.shape[1], frames.shape[0]))
    if frames.shape[2] == 1:
        # Grayscale video
        frames = np.concatenate((frames, frames, frames), axis=2)
        print(frames.shape)
    for i in range(frames.shape[3]):
        frame = frames[:, :, :, i]
        if is_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif is_ycrcb:
            frame = cv2.cvtColor(frame, cv2.COLOR_YCrCb2BGR)
        result.write(frame)
    result.release()


def simulate_moving_blob(M, N, L, n_frames, amp, n_periods, noise_amp):
    """
    Simulate a blob taking a random walk in the video as a source
    of missing data.  Change the video in place and return the mask
    of the data that's still there

    Parameters
    ----------
    M: int
        Height of image
    N: int
        Width of image
    L: int
        Blob width in pixels
    n_frames: int
        Number of frames
    amp: float
        Amplitude of oscillation
    n_periods: float
        Number of periods to go through
    noise: float
        Amount of noise to add in oscillatory trajectory
    Returns
    -------
    frames: ndarray(M, N, 3, n_frames)
        A video of a white square moving on a black background
    X: ndarray(n_frames, 2)
        Trajectory of the white square over time
    """
    from timeseries import get_random_walk_curve, smooth_curve
    X = get_random_walk_curve(100, int(n_frames/4), 2)
    X = smooth_curve(X, 4)
    X = X[0:min(X.shape[0], n_frames), :]
    if X.shape[0] < n_frames:
        diff = n_frames-X.shape[0]
        X = np.concatenate((X, np.flipud(X[-diff::, :])), axis=0)
    X -= np.min(X, axis=0)
    X /= np.max(X, axis=0)
    X *= noise_amp
    t = np.linspace(0, 2*np.pi*n_periods, n_frames)
    X[:, 0] += N/2 + amp*np.cos(t)
    X[:, 1] += M/2
    frames = np.zeros((M, N, 3, n_frames))
    XX, YY = np.meshgrid(np.arange(M), np.arange(N))
    XX = np.array([XX.flatten(), YY.flatten()]).T
    for i in range(n_frames):
        [v, u] = X[i, :]
        F = np.exp(-((XX[:, 0]-v)**2 + (XX[:, 1]-u)**2)/(2*L**2))
        frames[:, :, :, i] = np.reshape(F, (M, N, 1))
    frames = np.array(frames*255, dtype=np.uint8)
    return frames, X

def get_blob_cm(frames):
    """
    Parameters
    ----------
    frames: ndarray(M, N, ., n_frames)
        Video data
    
    Returns
    -------
    X: ndarray(n_frames, 2)
        Center of mass over all frames
    """
    M = frames.shape[0]
    N = frames.shape[1]
    XX, YY = np.meshgrid(np.arange(N), np.arange(M))
    n_frames = frames.shape[-1]
    if len(frames.shape) == 4:
        frames = np.sum(frames, axis=2)
    X = np.zeros((n_frames, 2))
    for i in range(n_frames):
        F = frames[:, :, i]
        denom = np.sum(F)
        X[i, 0] = np.sum(XX*F)/denom
        X[i, 1] = np.sum(YY*F)/denom
    return X

if __name__ == '__main__':
    M = 256
    N = 256
    L = 10
    n_frames = 300
    amp = 40
    n_periods = 10
    noise_amp = 20
    frames, X = simulate_moving_blob(M, N, L, n_frames, amp, n_periods, noise_amp)
    save_video("out.avi", frames, 30)