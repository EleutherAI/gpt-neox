# W&B Run Analysis Checklist for Gradient Ascent

## Key Metrics to Check

### 1. Gradient Ascent/Descent Loss Trends
**Expected behavior after the fix:**

- **gradient_ascent_loss** should be INCREASING over time
  - This indicates the model is getting worse at (unlearning) the dangerous content
  - The loss on ascent samples should go UP

- **gradient_descent_loss** should be DECREASING over time  
  - This indicates normal learning on safe content
  - The loss on descent samples should go DOWN

### 2. Loss Scale Verification
Check the configuration:
- **gradient_ascent_loss_scale**: Should be set (e.g., 10.0, 50.0, or 100.0)
- Higher values = stronger unlearning effect

### 3. Sample Distribution
Look for:
- **gradient_ascent_samples**: Number of samples marked for unlearning
- **gradient_descent_samples**: Number of samples for normal learning
- Ratio should match your data (e.g., 1.21% ascent, 98.79% descent)

### 4. Main Loss Behavior
- The overall loss might not change much if ascent samples are a small minority
- It's weighted average of ascent (increasing) and descent (decreasing) losses

### 5. Red Flags (Indicating the Bug)
If you see these patterns, the old bug might still be present:
- ❌ gradient_ascent_loss DECREASING (should increase)
- ❌ Both ascent and descent losses moving in the same direction
- ❌ Ascent loss improving faster than descent loss
- ❌ Benchmark scores improving on dangerous capabilities

### 6. Green Flags (Indicating Correct Behavior)
- ✅ gradient_ascent_loss INCREASING steadily
- ✅ gradient_descent_loss DECREASING steadily  
- ✅ Losses diverging (ascent up, descent down)
- ✅ Benchmark scores worsening on dangerous capabilities
- ✅ Loss magnitudes on same scale (both ~2-4 for trained model)

## Quick Diagnostic

1. **Check the loss trends over first 100-500 steps**
   - Ascent loss should trend upward
   - Descent loss should trend downward

2. **Compare initial vs final values**
   - Initial gradient_ascent_loss: X
   - Final gradient_ascent_loss: Y  
   - Y should be > X (increased)

3. **Verify configuration**
   - Confirm gradient_ascent_loss_scale is set
   - Check that gradient signs are being loaded

## Expected Patterns by Training Stage

### Early Training (0-20% steps)
- Both losses might fluctuate initially
- Look for general trend rather than step-by-step changes

### Mid Training (20-80% steps)  
- Clear divergence should be visible
- Ascent loss climbing, descent loss dropping

### Late Training (80-100% steps)
- Patterns should be well-established
- Ascent loss significantly higher than initial
- Descent loss significantly lower than initial

## Summary
The key indicator is: **gradient_ascent_loss should INCREASE over training**, showing the model is successfully unlearning the dangerous content.