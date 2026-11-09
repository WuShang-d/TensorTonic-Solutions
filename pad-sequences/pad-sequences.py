import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)

    ans = []
    for seq in seqs:
        seq = seq[:max_len]
        seq.extend([pad_value] * (max_len - len(seq)))
        ans.append(seq)
    
    return np.array(ans, dtype=int).reshape(len(seqs), max_len)
            