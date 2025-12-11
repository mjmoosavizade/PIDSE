# Normalized Physics Hypothesis Testing

## Experiment Overview

**Start Time:** November 25, 2025, 15:51:30
**Status:** RUNNING

## Configuration

Testing 3 physics types × 5 weights = **15 configurations**

### Hypotheses:
1. **Conservation** - Energy/momentum conservation laws
2. **Dynamics Consistency** - Physical dynamics constraints  
3. **Smoothness** - Trajectory smoothness constraints

### Physics Weights Tested:
- 0.0 (baseline - no physics)
- 0.01
- 0.05
- 0.1
- 0.5

### Key Innovation:
✅ **Loss Normalization ENABLED** - Physics loss automatically scaled to match estimation loss magnitude

## What's Different from Previous Tests?

| Aspect | Previous Tests | This Test |
|--------|---------------|-----------|
| Physics Loss Scale | Raw (273x larger) | Normalized (matched) |
| Physics Weights | 0.001 (tiny) | 0.01-0.5 (meaningful) |
| Scale Mismatch | 273x | ~1-10x |
| Training Duration | 30 epochs | 15 epochs (quick test) |
| Hypotheses | Single type | 3 types compared |

## Expected Outcomes

### Success Scenario:
- One or more configurations beat baseline ATE (2.907m)
- Physics loss contributes meaningfully without overwhelming
- Scale ratio stays in 1-10x range

### Learning Scenario:
- Normalization enables stable training
- Higher weights (0.1, 0.5) work without divergence
- Can identify which physics type works best for KITTI

## Monitoring Progress

```bash
# Watch live updates
tail -f experiments/normalized_hypothesis_testing_log.txt

# Check status
bash monitor_normalized_testing.sh

# View results when complete
cat experiments/normalized_hypothesis_results.json
```

## Estimated Timeline

- ~15 minutes per configuration
- 15 configurations total
- **Total: ~3-4 hours**

## Key Metrics to Watch

1. **Scale Ratio** - Should be ~1-10x (not 273x)
2. **Val ATE** - Target: < 2.907m (beat baseline)
3. **Training Stability** - No NaN losses, smooth convergence
4. **Test Performance** - Final generalization measure

## Results Will Show:

- Whether normalization makes physics constraints effective
- Optimal physics type for KITTI vehicle odometry
- Optimal physics weight with normalization
- Comparative analysis for research paper

---

**Next Steps After Completion:**
1. Analyze `normalized_hypothesis_results.json`
2. Compare best config vs baseline (2.907m ATE)
3. Generate figures for research paper
4. Document findings in paper methodology section
