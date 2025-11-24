# PIDSE Development Roadmap & TODOs

This document tracks ongoing development tasks, improvements, and future research directions for the PIDSE (Physics-Informed Differentiable State Estimator) framework.

---

## 📋 Current Status

**Framework Status:** ✅ Core implementation complete and validated  
**Latest Training:** KITTI odometry (30 epochs, ATE: 2.907m, RPE: 1.031m)  
**Last Updated:** October 22, 2025

---

## 🚀 High Priority - Performance Improvements

These improvements are expected to yield 20-40% performance gains with minimal effort.

### 1. Enable Physics Constraints
**Priority:** 🔴 HIGH  
**Effort:** Low  
**Expected Impact:** 15-25% improvement in long-term drift

**Tasks:**
- [ ] Set `physics_weight = 0.05` in training config
- [ ] Run training with physics loss enabled
- [ ] Compare ATE/RPE with baseline (physics_weight=0)
- [ ] Tune physics weight (try 0.01, 0.05, 0.1, 0.5)
- [ ] Document impact on trajectory consistency

**Rationale:** Current training has physics loss computed but weighted to 0. Enabling this should enforce physical constraints and reduce drift.

**File:** `experiments/kitti_full_training.py` line 438

---

### 2. Train for More Epochs
**Priority:** 🔴 HIGH  
**Effort:** Low (just time)  
**Expected Impact:** 10-20% improvement

**Tasks:**
- [ ] Increase epochs from 30 → 50
- [ ] Monitor for overfitting on validation set
- [ ] If no overfitting, try 100 epochs
- [ ] Implement early stopping based on validation loss
- [ ] Log convergence analysis

**Configuration:**
```python
'num_epochs': 50,  # or 100
```

**File:** `experiments/kitti_full_training.py` line 441

---

### 3. Larger Network Capacity
**Priority:** 🟡 MEDIUM  
**Effort:** Low  
**Expected Impact:** 10-15% improvement

**Tasks:**
- [ ] Increase hidden layers: `[32, 32]` → `[64, 64, 32]`
- [ ] Alternatively try: `[128, 64, 32]` or `[64, 64, 64, 32]`
- [ ] Monitor parameter count (target: ~15-20k params)
- [ ] Compare training stability
- [ ] Evaluate test performance improvement

**Configuration:**
```python
'pinn_hidden_layers': [64, 64, 32],  # Currently [32, 32]
```

**File:** `experiments/kitti_full_training.py` line 434

---

### 4. Add More Training Sequences
**Priority:** 🟡 MEDIUM  
**Effort:** Low  
**Expected Impact:** 15-20% better generalization

**Tasks:**
- [ ] Add sequences 06, 07, 08 to training set
- [ ] Keep sequences 03, 04 for testing (preserve test set)
- [ ] Monitor training time increase
- [ ] Evaluate cross-sequence generalization
- [ ] Create sequence-specific performance breakdown

**Current:**
```python
'train_sequences': ['00', '01', '02'],  # 408 samples
'val_sequences': ['05'],                 # 55 samples
'test_sequences': ['03', '04']           # 21 samples
```

**Proposed:**
```python
'train_sequences': ['00', '01', '02', '06', '07', '08'],  # ~816 samples
'val_sequences': ['05', '09'],                             # ~110 samples
'test_sequences': ['03', '04']                             # 21 samples (unchanged)
```

**File:** `experiments/kitti_full_training.py` lines 450-452

---

### 5. Compare with Baseline EKF
**Priority:** 🔴 HIGH  
**Effort:** Medium  
**Expected Impact:** Critical for publication

**Tasks:**
- [ ] Implement traditional EKF with manually tuned Q and R
- [ ] Run on same KITTI test sequences
- [ ] Compare: ATE, RPE, computation time
- [ ] Document tuning effort (hours spent)
- [ ] Create comparison plots
- [ ] Quantify improvement: (Baseline - PIDSE) / Baseline × 100%

**New Script:** `experiments/baseline_ekf_comparison.py`

**Comparison Metrics:**
- Trajectory error (ATE, RPE)
- Computational efficiency
- Tuning effort (person-hours)
- Robustness to noise
- Generalization across sequences

---

## 🔬 Medium Priority - Research Extensions

### 6. Optimize Learning Rate Schedule
**Priority:** 🟡 MEDIUM  
**Effort:** Low  
**Expected Impact:** 5-10% faster convergence

**Tasks:**
- [ ] Implement CosineAnnealingLR
- [ ] Try OneCycleLR with warmup
- [ ] Compare with current ReduceLROnPlateau
- [ ] Profile training time savings
- [ ] Document optimal schedule

**Example:**
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=1e-3,
    epochs=50,
    steps_per_epoch=len(train_loader)
)
```

---

### 7. Increase Batch Size & Sequence Length
**Priority:** 🟡 MEDIUM  
**Effort:** Medium (memory requirements)  
**Expected Impact:** Better gradient estimates

**Tasks:**
- [ ] Test batch_size: 2 → 4 → 8
- [ ] Test sequence_length: 50 → 75 → 100
- [ ] Monitor GPU memory usage
- [ ] Evaluate training stability
- [ ] Measure impact on convergence speed

**Current Constraints:**
- GPU memory limit
- Sequence length affects EKF computational cost

---

### 8. Data Augmentation
**Priority:** 🟢 LOW  
**Effort:** Medium  
**Expected Impact:** Better robustness

**Tasks:**
- [ ] Add Gaussian noise to measurements
- [ ] Random dropout of sensor readings
- [ ] Temporal subsampling
- [ ] Test robustness improvements
- [ ] Validate on corrupted test data

---

## 📊 Advanced Experiments

### 9. Ablation Studies
**Priority:** 🟡 MEDIUM (for publication)  
**Effort:** High  
**Expected Impact:** Scientific validation

**Studies to Conduct:**
1. **Learnable vs Fixed Noise Matrices**
   - [ ] Train with fixed Q and R (current best values)
   - [ ] Compare with learnable Q and R
   - [ ] Quantify benefit of learning

2. **Physics Loss Contribution**
   - [ ] Train without physics loss (current baseline)
   - [ ] Train with increasing physics weights
   - [ ] Plot performance vs physics_weight
   - [ ] Find optimal balance

3. **Network Architecture**
   - [ ] Test different activation functions (tanh, ReLU, SiLU)
   - [ ] Test with/without skip connections
   - [ ] Vary network depth and width

4. **State Space Dimensionality**
   - [ ] Simplified 6D state [px, py, pz, vx, vy, vz]
   - [ ] Compare with full 12D state
   - [ ] Evaluate trade-off

**Output:** `docs/ablation_studies.md`

---

### 10. Multi-Dataset Training
**Priority:** 🟢 LOW  
**Effort:** High  
**Expected Impact:** Better generalization

**Tasks:**
- [ ] Add EuRoC MAV dataset support
- [ ] Implement multi-dataset training loop
- [ ] Test cross-dataset generalization
- [ ] Evaluate domain adaptation
- [ ] Create unified evaluation protocol

**Datasets:**
- KITTI (outdoor vehicle)
- EuRoC (indoor/outdoor MAV)
- TUM RGB-D (indoor handheld)

---

### 11. Architecture Variants
**Priority:** 🟢 LOW  
**Effort:** High  
**Expected Impact:** Research exploration

**Experiments:**
1. **Attention Mechanisms**
   - [ ] Add self-attention to PINN
   - [ ] Test temporal attention
   - [ ] Evaluate computational cost

2. **Recurrent Components**
   - [ ] Replace MLP with LSTM/GRU
   - [ ] Compare memory usage
   - [ ] Test on longer sequences

3. **Alternative Activations**
   - [ ] Swish, GELU, ELU
   - [ ] Compare convergence
   - [ ] Profile training speed

---

## 🐛 Bug Fixes & Maintenance

### Known Issues
- [ ] None currently identified

### Code Quality
- [ ] Add docstrings to all public methods
- [ ] Increase test coverage to 80%
- [ ] Add type hints throughout
- [ ] Profile and optimize bottlenecks
- [ ] Update README with latest results

---

## 📝 Documentation Tasks

### High Priority
- [x] Create RESULTS_SUMMARY.md with KITTI results
- [ ] Update README.md with quick start guide
- [ ] Add installation instructions for dependencies
- [ ] Create API documentation
- [ ] Write tutorial notebooks

### Medium Priority
- [ ] Document learned noise matrix interpretation
- [ ] Create visualization guide
- [ ] Add troubleshooting section
- [ ] Document hyperparameter tuning strategies

### Low Priority
- [ ] Create project website
- [ ] Record demo videos
- [ ] Write blog post about results

---

## 🚀 Deployment & Production

### Immediate
- [ ] Clean up workspace (remove test files)
- [ ] Update .gitignore
- [ ] Create requirements.txt freeze
- [ ] Tag release v0.1.0

### Future
- [ ] Create Docker container
- [ ] Add continuous integration (GitHub Actions)
- [ ] Implement model versioning
- [ ] Create REST API for inference
- [ ] Optimize for real-time inference

---

## 📅 Timeline Estimates

### Week 1-2 (Quick Wins)
- Enable physics constraints
- Train for 50-100 epochs
- Larger network architecture
- Add more training sequences

**Expected Outcome:** 30-50% performance improvement

### Week 3-4 (Baseline Comparison)
- Implement traditional EKF
- Run comprehensive comparisons
- Document results for publication

**Expected Outcome:** Quantified improvement over baseline

### Week 5-8 (Advanced Experiments)
- Ablation studies
- Optimization experiments
- Multi-dataset evaluation

**Expected Outcome:** Publication-ready results

---

## 📊 Success Metrics

### Performance Targets
- **ATE:** < 2.0m (current: 2.907m) - 31% improvement needed
- **RPE:** < 0.8m (current: 1.031m) - 22% improvement needed
- **Drift:** < 2.0m (current: 2.906m) - 31% improvement needed

### Research Targets
- [ ] Demonstrate 20-30% improvement over manual EKF tuning
- [ ] Publish results in robotics/ML conference
- [ ] Release open-source package
- [ ] Achieve 100+ GitHub stars

---

## 🔗 Related Issues & PRs

- Issue tracking: (To be created)
- Feature requests: (To be created)
- Bug reports: (To be created)

---

## 📞 Contact & Collaboration

For questions or collaboration inquiries:
- GitHub: mjmoosavizade/PIDSE
- Email: (To be added)

---

**Last Updated:** October 22, 2025  
**Next Review:** Weekly updates after each major milestone
