
1. GAT model: 2 layers, hidden size 8 (uncertain), heads per layer [8, 1], output = 7 classes.  
2. Input dropout 0.6 (uncertain), attention (coefficient) dropout 0.6 (uncertain).  
3. Learning rate 0.01 (uncertain), L2 weight decay 5e‑4 (uncertain), optimizer Adam (uncertain).  
4. Batch size = full graph (2485 nodes) (uncertain), no mini‑batching.  
5. Max epochs 100 000, early stopping with GATCriterion: stop if neither validation loss nor validation accuracy improves for 100 epochs (patience = 100) (uncertain).  
6. Train/val/test split: 20 labeled nodes per class for train (140), 30 per class for val (210), remaining nodes for test (2135).  
7. Features: raw binary bag‑of‑words [2485, 1433] (no normalization) (uncertain).  
8. Graph preprocessing: add self‑loops, keep undirected largest connected component (already provided).  
9. Weight‑initialisation seed read from env SEED; set `numpy.random.seed(SEED)` and `torch.manual_seed(SEED)`.  
10. No learning‑rate decay (`lr_decay_factor = 0.0`) and no alternating optimization.  
11. Loss: categorical cross‑entropy; evaluation metric: accuracy.  
12. After training, predict class for every node in `test_idx` and write `/output/predictions.csv` (`id,prediction`).  
13. Expected runtime < 5 minutes on CPU.  
14. All values marked **(uncertain)** are those not explicitly given in the paper or supplied configs.