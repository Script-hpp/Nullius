

1. Model: two‑layer GCN (input → hidden = 16 → output = 6 classes) with ReLU after the first layer and linear activation after the second.  
2. Dropout rate = 0.5 applied to both layers (controlled by a placeholder during training).  
3. Weight decay (L2) = 5·10⁻⁴ applied only to the weights of the first GraphConvolution layer.  
4. Optimizer = Adam, learning rate = 0.01.  
5. Input features are row‑normalized (each row sums to 1) before feeding the network.  
6. Adjacency matrix is augmented with self‑loops and symmetrically normalized: Â = D̃⁻¹ᐟ² (A + I) D̃⁻¹ᐟ².  
7. Weight initialization follows Glorot & Bengio (2010) – the default TensorFlow vari‑able initializer.  
8. Training is full‑batch (all nodes processed each epoch), max epochs = 200, early stopping patience = 10 (stop if validation loss does not improve for 10 consecutive epochs).  
9. Validation set = 500 labeled nodes (provided); its labels are masked out and **not** used in the training loss.  
10. Random seed: read env var SEED, set `numpy.random.seed(SEED)` and `torch.manual_seed(SEED)` (and `tf.set_random_seed(SEED)` if needed).  
11. Loss = masked softmax cross‑entropy + weight‑decay term on first‑layer weights.  
12. Evaluation metric = classification accuracy on the test set.  
13. (Uncertain) Batch size: full‑batch (entire graph) – not explicitly stated in the paper but implied by the code.  
14. (Uncertain) Number of runs for reporting: paper reports mean accuracy over 100 random initializations; this plan executes a single run.  
15. (Uncertain) Test set size = 1000 nodes (as given in the provided split).