
class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction



def phi(x: torch.Tensor, y: torch.Tensor, dim: int, tau: float) -> torch.Tensor:
    """Compute the cosine similarity between two tensors.

    Args:
        x (torch.Tensor): The first tensor.
        y (torch.Tensor): The second tensor.
        dim (int): The dimension to compute the cosine similarity.
        tau (float): The temperature parameter.

    Returns:
        torch.Tensor: The cosine similarity between the two tensors.
    """
    return torch.exp(F.cosine_similarity(x, y, dim=dim) / tau)


class InfoNCELoss(nn.Module):
    """InfoNCE loss function for contrastive learning."""
    __constants__ = ['reduction']

    def __init__(self, reduction='mean'):
        """Initialize the loss function."""
        super().__init__(reduction=reduction)

    def forward(self, query, positive, negatives):
        """Compute the loss.

        Args:
            query (torch.Tensor): The query embedding.
            positive (torch.Tensor): The positive document embedding.
            negatives (torch.Tensor): The negative document embedding.

        Returns:
            torch.Tensor: The loss value.
        """
        return info_nce(query,
                        positive,
                        negatives,
                        1, self.reduction)
        # TODO: check dimension through triplet loss, may be similar


def info_nce(queries: torch.Tensor, positives: torch.Tensor,
             negatives: torch.Tensor, temp: float, reduction: str) -> torch.Tensor:
    """Compute the InfoNCE loss.

    Args:
        queries (torch.Tensor): The query tensor.
        positives (torch.Tensor): The positive tensor.
        negatives (torch.Tensor): The negative tensor.
        temp (float): The temperature parameter.
        reduction (str): The reduction method.

    Returns:
        torch.Tensor: The InfoNCE loss.
    """
    numerator_logits = F.cosine_similarity(queries, positives, dim=1)  # B
    numerator_vals = torch.exp(numerator_logits / temp)
    denominator = torch.cat((positives, negatives), dim=0)
    denominator_logits = F.cosine_similarity(queries.unsqueeze(1),
                                             denominator,
                                             dim=2)  # (B, Bx2)
    denominator_vals = torch.sum(torch.exp(denominator_logits / temp),
                                 dim=-1)  # (B
    losses = -torch.log(numerator_vals / denominator_vals)
    if reduction == 'mean':
        return losses.mean()
    else:
        raise ValueError(f"Loss reduction {reduction} not supported.")


class TripletMarginLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given an input
    
    Args:
        margin (float, optional): Default: :math:`1`.
        p (int, optional): The norm degree for pairwise distance. Default: :math:`2`.
        eps (float, optional): Small constant for numerical stability. Default: :math:`1e-6`.
        swap (bool, optional): The distance swap is described in detail in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. Default: ``False``.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, D)` or :math:`(D)` where :math:`D` is the vector dimension.
        - Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'`` and
          input shape is :math:`(N, D)`; a scalar otherwise.

    Examples::

    >>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    >>> anchor = torch.randn(100, 128, requires_grad=True)
    >>> positive = torch.randn(100, 128, requires_grad=True)
    >>> negative = torch.randn(100, 128, requires_grad=True)
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.bmva.org/bmvc/2016/papers/paper119/index.html
    """
    __constants__ = ['margin', 'p', 'eps', 'swap', 'reduction']
    margin: float
    p: float
    eps: float
    swap: bool

    def __init__(self, margin: float = 1.0, p: float = 2., eps: float = 1e-6, swap: bool = False, size_average=None,
                 reduce=None, reduction: str = 'mean'):
        super().__init__(size_average, reduce, reduction)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return x(anchor, positive, negative, margin=self.margin, p=self.p,
                                     eps=self.eps, swap=self.swap, reduction=self.reduction)

