import numpy as np

def smooth_mode(y_pred: np.ndarray, k: int, n_classes: int) -> np.ndarray:
    """Slankiojanti moda langelyje k (k nelyginis)."""
    y = np.asarray(y_pred)
    n = len(y); h = k // 2
    out = np.empty(n, dtype=y.dtype)
    for i in range(n):
        a = max(0, i - h)
        b = min(n, i + h + 1)
        window = y[a:b]
        counts = np.bincount(window, minlength=n_classes)
        out[i] = counts.argmax()
    return out

def apply_class_biases(logits: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """Prideda vektorių (C,) prie kiekvienos logit eilutės prieš argmax."""
    return logits + biases.reshape(1, -1)

