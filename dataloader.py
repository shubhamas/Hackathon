import torch
from torch.utils.data import Dataset

# Custom Dataset for N-MNIST .bin files
class NMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Support both 0-9 and class names
        self.classes = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            for f in os.listdir(class_dir):
                if f.endswith(".bin"):
                    self.samples.append(
                        (os.path.join(class_dir, f), self.class_to_idx[cls_name])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with open(path, "rb") as f:
            raw_bytes = f.read()

        # Parse events (5 bytes per event)
        # Format based on inspection: x(1), y(1), p/ts(1), ts(1), ts(1)
        data = np.frombuffer(raw_bytes, dtype=np.uint8)

        num_events = len(data) // 5
        data = data[:num_events * 5].reshape(-1, 5)

        x = data[:, 0]
        y = data[:, 1]
        p = (data[:, 2] >> 7)   # Polarity in MSB of byte 2

        print("x:", x.shape, "y:", y.shape, "p:", p.shape)

        # Create frame (2, 34, 34)
        frame = np.zeros((2, 34, 34), dtype=np.float32)

        # Filter valid coordinates
        mask_valid = (x < 34) & (y < 34)
        x = x[mask_valid]
        y = y[mask_valid]
        p = p[mask_valid]

        # Accumulate events into frame
        # Use (y, x) for histogram to map to (row, col) (Height, Width)
        mask_on = (p == 1)
        mask_off = (p == 0)

        # Channel 0: OFF, Channel 1: ON
        if np.any(mask_off):
            hist_off, _, _ = np.histogram2d(
                y[mask_off], x[mask_off],
                bins=[34, 34],
                range=[[0, 34], [0, 34]]
            )
            frame[0] = hist_off

        if np.any(mask_on):
            hist_on, _, _ = np.histogram2d(
                y[mask_on], x[mask_on],
                bins=[34, 34],
                range=[[0, 34], [0, 34]]
            )
            frame[1] = hist_on

        # Normalize
        if frame.max() > 0:
            frame = frame / frame.max()

        return torch.from_numpy(frame), label
