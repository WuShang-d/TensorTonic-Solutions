import torch

def train_tiny_micrograd_mlp(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    learning_rate: float,
    steps: int
) -> tuple:

    W = [w.clone() for w in weights]
    b = [bias.clone() for bias in biases]

    update_loss_history = []

    def forward():
        H = inputs
        activations = [H]

        for w, bias in zip(W, b):
            H = torch.tanh(H @ w.T + bias)
            activations.append(H)

        return activations

    for _ in range(steps):
        # 1. Forward
        activations = forward()
        pred = activations[-1].squeeze(-1)

        # 2. Loss
        loss = ((pred - targets) ** 2).sum()
        update_loss_history.append(loss)

        # 3. Backward
        grad = (2 * (pred - targets)).unsqueeze(-1)

        dW = [None] * len(W)
        db = [None] * len(b)

        for i in range(len(W) - 1, -1, -1):
            H = activations[i + 1]
            H_prev = activations[i]

            delta = grad * (1 - H ** 2)

            dW[i] = delta.T @ H_prev
            db[i] = delta.sum(dim=0)

            grad = delta @ W[i]

        # 4. Gradient descent
        for i in range(len(W)):
            W[i] = W[i] - learning_rate * dW[i]
            b[i] = b[i] - learning_rate * db[i]

    # 5. Final forward
    activations = forward()
    final_pred = activations[-1].squeeze(-1)
    final_loss = ((final_pred - targets) ** 2).sum()

    return final_pred, final_loss, W, b, update_loss_history