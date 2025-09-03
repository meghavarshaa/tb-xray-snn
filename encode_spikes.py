import numpy as np
import os

def poisson_spike_encoding_batch(images, time_bins=100, max_rate=100, batch_size=100):
    """
    Encode images into Poisson spike trains in batches to reduce memory load.
    
    Parameters:
    - images: np.ndarray, shape (N, H, W, 1), normalized [0,1]
    - time_bins: int, number of discrete time steps
    - max_rate: int, max firing rate in Hz
    - batch_size: int, number of images to process per batch
    
    Returns:
    - spike_trains: np.ndarray, shape (N, time_bins, H*W), dtype uint8
    """
    N, H, W, C = images.shape
    assert C == 1, "Images must have single channel"

    spike_trains_batches = []
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = images[start:end]
        batch_flat = batch.reshape(end - start, H * W)
        
        spike_prob = batch_flat * (max_rate / 1000)

        batch_spikes = np.random.rand(end - start, time_bins, H * W) < spike_prob[:, np.newaxis, :]
        spike_trains_batches.append(batch_spikes.astype(np.uint8))
        
        print(f"Encoded batch: {start} – {end} out of {N}")

    spike_trains = np.concatenate(spike_trains_batches, axis=0)
    return spike_trains


def main():
    data_dir = "./"
    train_data_path = os.path.join(data_dir, "X_train.npy")
    test_data_path = os.path.join(data_dir, "X_test.npy")

    print("Loading preprocessed training data...")
    X_train = np.load(train_data_path)  # (3360, 224, 224, 1)
    print("Loading preprocessed test data...")
    X_test = np.load(test_data_path)    # (840, 224, 224, 1)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    time_bins = 100
    max_rate = 100
    batch_size = 100  # Adjust batch size if memory issues continue

    print("Encoding training images to spike trains with batching...")
    train_spikes = poisson_spike_encoding_batch(X_train, time_bins, max_rate, batch_size)
    print("Encoding test images to spike trains with batching...")
    test_spikes = poisson_spike_encoding_batch(X_test, time_bins, max_rate, batch_size)
    
    print(f"Train spikes shape: {train_spikes.shape}")
    print(f"Test spikes shape: {test_spikes.shape}")

    np.save(os.path.join(data_dir, "train_spikes.npy"), train_spikes)
    np.save(os.path.join(data_dir, "test_spikes.npy"), test_spikes)

    print("Spike train encoding complete and saved.")


if __name__ == "__main__":
    main()
