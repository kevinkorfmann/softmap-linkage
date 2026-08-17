# SoftMap Rust accelerator

This optional extension accelerates the two pairwise-recombination kernels. The
Python package remains fully functional without it and automatically falls back
to NumPy.

For local development:

```bash
uv pip install ./rust
```

Set `SOFTMAP_DISABLE_RUST=1` to force the reference NumPy implementation for
equivalence testing.
