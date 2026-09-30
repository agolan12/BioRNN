"""
Two remedies for vanishing/exploding gradients from Pascanu, Mikolov & Bengio
(2013), "On the difficulty of training recurrent neural networks":

  - Gradient norm clipping (their Algorithm 1): if the L2 norm of the full
    gradient exceeds a threshold, rescale the whole gradient down to that
    threshold before the optimizer step. Targets exploding gradients by
    bounding every single parameter update. Used directly via
    `torch.nn.utils.clip_grad_norm_` -- no custom code needed.

  - `norm_preserving_regularizer` (their Sec. 3.2, Eq. 16): a soft penalty
    added to the loss that discourages the backpropagated error signal from
    changing norm as it moves one step back in time. Pushing the recurrent
    Jacobian toward norm-preserving directly counteracts vanishing
    gradients, rather than just capping large ones.
"""

import torch


def norm_preserving_regularizer(loss, hidden_states, eps=1e-6):
    """
    Exact for tasks with a single readout at the final timestep (e.g. the
    Adding Problem), where dL/dh_t = (dL/dh_{t+1}) @ (dh_{t+1}/dh_t) with no
    other path from h_t to the loss -- so two consecutive backpropagated
    gradients are related by exactly one Jacobian-vector product, and their
    norm ratio is exactly the quantity Eq. 16 wants pushed toward 1.

    For tasks with a readout at every timestep (e.g. the Copy Task), dL/dh_t
    also receives a direct contribution from that timestep's own output, so
    the ratio is only an approximation of the paper's quantity -- but it's
    still a reasonable heuristic that penalizes large swings in backpropagated
    gradient norm.

    `hidden_states` must be the list of hidden-state tensors returned by
    `model(x, return_hidden_states=True)`; the model computed them with
    `requires_grad=True`, so they're valid tensors to differentiate through.
    `create_graph=True` keeps this differentiable a second time, so that
    backpropagating the regularizer term contributes to the parameter
    gradients exactly like any other loss term.
    """
    grads = torch.autograd.grad(loss, hidden_states, create_graph=True, retain_graph=True)
    omega = 0.0
    for earlier, later in zip(grads[:-1], grads[1:]):
        ratio = earlier.norm(dim=1) / (later.norm(dim=1) + eps)
        omega = omega + ((ratio - 1.0) ** 2).mean()
    return omega / (len(grads) - 1)
