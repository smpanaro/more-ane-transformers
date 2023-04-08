import warnings
from contextlib import contextmanager

def _gpt2_warning_filter():
    patterns = [
        # TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # if this fails the trace will fail
        (r".*Converting a tensor to a Python boolean might cause the trace to be incorrect.*", [188]),

        # UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
        #   k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
        #   q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
        #   v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # C and n_head will never be negative
        (r".*__floordiv__ is deprecated.*", [67, 68, 69]),

        # TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # k will always be the same size
        (r".*Converting a tensor to a Python float.*", [77])
    ]
    for pattern, linenos in patterns:
        for lineno in linenos:
            warnings.filterwarnings("ignore", pattern, lineno=lineno)


@contextmanager
def silence_known_trace_warnings(model_name: str):
    """
    Silence known and safe to ignore warnings.
    """
    with warnings.catch_warnings():
        if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
            _gpt2_warning_filter()
        yield
