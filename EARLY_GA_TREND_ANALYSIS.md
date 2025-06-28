# Early Training GA Trend Analysis (Steps 0-570)

## Critical Finding: Initial Increase Followed by Decrease

Looking at the first 570 training steps reveals a crucial pattern that changes our understanding:

### The Two-Phase Pattern

1. **Phase 1 (Steps 11-88): INCREASING Loss** ✓
   - First GA loss: 1.895
   - Peak GA loss: **3.016 at step 88**
   - This is gradient ascent working correctly!

2. **Phase 2 (Steps 88-570): DECREASING Loss** ✗
   - After the peak, loss starts declining
   - Ends at 1.891 (below starting point)
   - This suggests gradient ascent is failing

### Detailed Early Step Analysis

**First 5 GA steps show correct behavior:**
- Step 11: 1.895
- Step 33: 2.207 (↑ 16.5%)
- Step 44: 2.100 (slight dip)
- Step 55: 2.224 (↑ recovery)
- Step 66: 2.125

**The peak at step 88 (loss = 3.016) represents a 59% increase from start!**

### What This Tells Us

1. **Gradient Ascent Initially Works**
   - The loss correctly increases for the first ~80 steps
   - The model is successfully getting worse at the GA task
   - This proves the implementation CAN work

2. **Something Changes Around Step 88**
   - After reaching peak loss of 3.016, the trend reverses
   - Loss begins steadily decreasing
   - By step 570, we're back near the starting point

### Possible Explanations for the Reversal

1. **Learning Rate Decay**
   - The learning rate schedule might be decreasing too aggressively
   - With `ga_lr_scale=0.5`, the effective GA learning rate becomes too small
   - The model's internal representations solidify, resisting further GA

2. **Gradient Interference**
   - The 1:10 ratio means 10 gradient descent steps between each GA step
   - Early in training, GA can overcome this
   - As the model learns, GD steps dominate and "repair" GA damage

3. **Representation Consolidation**
   - Early in training, model representations are fluid
   - GA can successfully disrupt learning
   - As training progresses, representations stabilize and resist GA

4. **Catastrophic Forgetting in Reverse**
   - The model might be "remembering" the GA dataset despite GA
   - Regular training on the main dataset reinforces shared features
   - GA becomes less effective as these features strengthen

### Segment Analysis Confirms the Pattern

- **Segment 1 (early)**: mean=2.180, std=0.244 (high variance, increasing)
- **Segment 2 (middle)**: mean=2.123, std=0.191 (stabilizing)
- **Segment 3 (late)**: mean=1.970, std=0.067 (decreasing, low variance)

The decreasing standard deviation shows the GA loss is becoming more stable but in the wrong direction.

### Comparison with Regular Training Loss

The regular training loss shows:
- Large spikes early (normal for early training)
- Steady decrease overall
- GA loss tracks similarly after step 88, suggesting GA is behaving like regular training

## Conclusions

1. **The implementation works initially** - GA successfully increases loss for ~80 steps
2. **A transition occurs around step 88** where GA effectiveness breaks down
3. **The decreasing loss after step 88** indicates GA is failing to achieve its objective
4. **This is likely due to**:
   - Learning rate becoming too small
   - Model representations becoming too stable
   - Interference from interleaved gradient descent

## Implications

This pattern suggests:
- **Short bursts of GA might be more effective** than sustained interleaved training
- **Higher GA learning rates** might be needed as training progresses
- **The current approach fails** to maintain unlearning throughout training
- **Early training GA works**, but the method doesn't scale to longer training

The fact that GA initially works but then fails is actually more concerning than if it never worked at all - it suggests the unlearning is temporary and easily reversed.