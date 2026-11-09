import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    pos   = np.arange(0, seq_len)[:, None]          # (T, 1)
    group = np.arange(0, d_model, 2)                # 奇数 d 时多一组，无所谓
    theta = base ** (-group / d_model)
    omega = pos * theta                             # (T, ceil(d/2))

    PE = np.empty((seq_len, d_model), dtype=np.float64)
    PE[:, 0::2] = np.sin(omega)                     # 偶数位：正好 ceil(d/2) 个槽
    PE[:, 1::2] = np.cos(omega)[:, : d_model // 2]  # 奇数位：只有 floor(d/2) 个槽，砍掉多余列
    return PE