"""
Modified cross_entropy with amplified gradient ascent for testing.
This makes the gradient ascent effect 10x stronger to overcome the sample imbalance.
"""

import torch
from .cross_entropy import _VocabParallelCrossEntropy as _VocabParallelCrossEntropyOriginal

class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, sample_signs=None):
        # Use original forward
        loss = _VocabParallelCrossEntropyOriginal.forward(ctx, vocab_parallel_logits, target, sample_signs=None)
        
        # Store amplified signs for backward
        if sample_signs is not None:
            # Amplify gradient ascent samples by 10x
            amplified_signs = sample_signs.clone()
            ascent_mask = sample_signs < 0
            amplified_signs[ascent_mask] = -10.0  # 10x amplification for ascent
            ctx.sample_signs = amplified_signs
        else:
            ctx.sample_signs = None
            
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        # Get original gradients
        grad_input, _, _ = _VocabParallelCrossEntropyOriginal.backward(ctx, grad_output)
        
        # Apply amplified signs
        if hasattr(ctx, 'sample_signs') and ctx.sample_signs is not None:
            # Create a version of grad_output with amplified signs
            amplified_grad_output = grad_output * ctx.sample_signs
            # Recompute grad_input with amplification
            grad_input.mul_(amplified_grad_output.unsqueeze(dim=-1) / grad_output.unsqueeze(dim=-1))
        
        return grad_input, None, None

def vocab_parallel_cross_entropy(vocab_parallel_logits, target, sample_signs=None):
    """Helper function for the cross entropy with amplification."""
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, sample_signs)