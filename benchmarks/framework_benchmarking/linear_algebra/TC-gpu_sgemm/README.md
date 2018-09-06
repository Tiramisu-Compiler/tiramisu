Run `python sgemm.py` to benchmark with autotuned options. Requires tensori\_comprehensions library.

On a single K80 with current options 3072x3072 multiplication runs in 320ms including data copies to-from GPU.
