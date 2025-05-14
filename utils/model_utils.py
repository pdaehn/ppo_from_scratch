import torch.nn as nn

_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: list[int],
    activation: str = "tanh",
    output_activation: str | None = None,
) -> nn.Sequential:
    """
    Build a multi-layer perceptron (MLP) with specified input and output dimensions,
    hidden layer sizes, and activation functions.

    Args:
        input_dim: dimension of the input layer.
        output_dim: dimension of the output layer.
        hidden_sizes: list of integers specifying the sizes of hidden layers.
        activation: activation function to use for hidden layers.
        output_activation: activation function to use for the output layer.
    """
    act_cls = _ACTIVATIONS[activation]
    out_act_cls = _ACTIVATIONS.get(output_activation, None)
    dims = [input_dim, *hidden_sizes, output_dim]
    layers: list[nn.Module] = []

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(hidden_sizes):
            layers.append(act_cls())
        elif out_act_cls is not None:
            layers.append(out_act_cls())

    return nn.Sequential(*layers)
