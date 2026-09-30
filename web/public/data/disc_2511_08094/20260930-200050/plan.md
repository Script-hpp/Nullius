1. Architecture: SL-GCN per equations; X⁽⁰⁾=features, Y⁽⁰⁾=0; F_θ = GCNConv (ÂXW) with Â=D⁻¹ᐟ²(A+I)D⁻¹ᐟ² (uncertain self-loops); σ=ReLU (uncertain).
2. Depths: 7 fixed depths ∈ {2,4,8,16,32,64,128} (uncertain exact set); hidden_dim=64 (uncertain).
3. ODE params: dt=1.0, α=0.1, γ=0.1 (uncertain, typical Stuart-Landau).
4. Classifier: Linear(hidden_dim→7) on final X⁽ᴸ⁾.
5. Loss: CrossEntropy on train_idx (140 nodes).
6. Optimizer: Adam lr=0.01, weight_decay=5e-4 (uncertain).
7. Training: Max 1000 epochs; early stopping on val_idx accuracy patience=100 (uncertain).
8. Selection: Best depth/model by val_idx accuracy across 7 depths.
9. Seed: numpy/torch.manual_seed(int(os.environ["SEED"])) for weight init.
10. Data: Load /data/cora_lcc_public.npz; use provided train_idx/val_idx/test_idx; features binary unnormalized (uncertain row-norm).
11. Inference: Predict test_idx (2135 nodes); write /output/predictions.csv header `id,prediction`.
12. Target: Paper reports 82.92% mean over 10 splits × 10 inits (uncertain: "100 splits" vs "10 splits" in quotes).
13. Compute: CPU only; ensure <5 min runtime.
14. Implementation: Pure PyTorch (no PyG) for speed; sparse matmul for ÂX.
15. No external data; single split file determines indices.