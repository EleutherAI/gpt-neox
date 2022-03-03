# GPT NeoX

new implementation by Kevin Ko.

1. Replace all the parallelization related code to OSLO.
2. Don't overuse `neox_args` everywhere. all the functions should be independent.
3. Ninja makes recompiling very fast. There is no need to precompile custom kernels.
4. Refactor all the dataset related code.
5. Add some useful CUDA kernels
