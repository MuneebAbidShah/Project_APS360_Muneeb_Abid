"""Model complexity for the report: layer-by-layer shapes and parameters, plus MACs per word."""
import torch

from src.models.crnn import count_parameters


def layer_table(model, height=32, width=128):
    """One row per stage of the CRNN, measured with a real forward pass on a 1×height×width image.
    Output shapes are written per image (batch dimension dropped); W is the input width."""
    rows, shapes = [], {}
    stages = [(f"conv{i + 1}", blk) for i, blk in enumerate(model.cnn)] + [("rnn", model.rnn), ("fc", model.fc)]
    hooks = [m.register_forward_hook(lambda m, i, o, k=k: shapes.__setitem__(k, tuple((o[0] if isinstance(o, tuple) else o).shape)))
             for k, m in stages]
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, 1, height, width, device=dev))
    for h in hooks:
        h.remove()
    model.train(was_training)

    def wexpr(w_out):
        d = width // w_out
        return "W" if d == 1 else f"W/{d}"

    rows.append({"layer": "Input", "details": "grayscale word image", "output_shape": f"1×{height}×W", "parameters": 0})
    for i, blk in enumerate(model.cnn):
        conv = blk[0]
        pool = [m for m in blk if isinstance(m, torch.nn.MaxPool2d)]
        p = f", max-pool {pool[0].kernel_size[0]}×{pool[0].kernel_size[1]}" if pool else ""
        _, c, h, w = shapes[f"conv{i + 1}"]
        rows.append({"layer": f"Conv block {i + 1}",
                     "details": f"3×3 conv {conv.in_channels}→{conv.out_channels}, BN, ReLU{p}",
                     "output_shape": f"{c}×{h}×{wexpr(w)}",
                     "parameters": count_parameters(blk)})
    T, _, f = shapes["rnn"]
    c_last = model.rnn.input_size
    rows.append({"layer": "Reshape", "details": "columns become time steps (T = W/4)",
                 "output_shape": f"T×{c_last}", "parameters": 0})
    rnn = model.rnn
    rows.append({"layer": "BiLSTM", "details": f"{rnn.num_layers} layers, {rnn.hidden_size} per direction, dropout {model.dropout.p:g}",
                 "output_shape": f"T×{f}", "parameters": count_parameters(rnn)})
    rows.append({"layer": "Linear + log-softmax", "details": f"{model.fc.in_features}→{model.fc.out_features} (78 chars + CTC blank)",
                 "output_shape": f"T×{model.fc.out_features}", "parameters": count_parameters(model.fc)})
    total = count_parameters(model)
    assert total == sum(r["parameters"] for r in rows), "layer table must add up to the model total"
    return rows, total


def macs_per_word(model, height=32, width=128):
    """Multiply-accumulates for one word image (analytical: convs, LSTM gates, linear layer).
    BatchNorm, ReLU, pooling and softmax are negligible and not counted."""
    macs, h, w = 0, height, width
    for blk in model.cnn:
        conv = blk[0]
        macs += conv.out_channels * conv.in_channels * 9 * h * w      # padding 1: output = input size
        for m in blk:
            if isinstance(m, torch.nn.MaxPool2d):
                h //= m.kernel_size[0]; w //= m.kernel_size[1]
    T, rnn = w, model.rnn
    for layer in range(rnn.num_layers):
        n_in = rnn.input_size if layer == 0 else 2 * rnn.hidden_size
        macs += 2 * T * 4 * rnn.hidden_size * (n_in + rnn.hidden_size)  # 2 directions, 4 gates
    macs += T * model.fc.in_features * model.fc.out_features
    return macs
