# ICC-LLM-ESR
Item Cluster Constrained  Large Language Models Enhancement for  Long-tailed Sequential Recommendation

This repository extends the work from [LLM-ESR](https://github.com/Applied-Machine-Learning-Lab/LLM-ESR) by incorporating clustering-based constraints to enhance the recommendation performance.

## Key Modifications

### 1. Clustering Implementation
- Added clustering functionality in `data/{dataset}/cluster.ipynb`
- Generates two key files:
  - `data/{dataset}/handled/item_cluster_labels.pkl` - Stores clustering labels for each item
  - `cluster_centers_8d.pkl` - Contains 8-dimensional cluster centers

### 2. Cluster Handling Utility
- New `ClusterHandler` class in `models/utils.py`
  - Loads cluster labels and centers, converting them to tensors
  - Maps items to their corresponding d-dimensional cluster centers (means)
  - Computes clustering constraint losses for non-noise items
  - Special handling for BERT-based models to:
    - Avoid index out-of-bounds errors during runtime
    - Prevent meaningless gradients
    - Generate static mapping table `item_cluster.pt`

### 3. Integration with LLM-ESR
- Modified `models/LLMESR.py` to:
  - Integrate `ClusterHandler`
  - Calculate beta values
  - Incorporate clustering constraint losses (with special handling for BERT models)

### 4. Experiment Configuration
- Updated experiment scripts in `experiment/*.bash` to include gamma parameter sweeps:
  ```bash
  --gamma 0.01 0.05 0.1 0.5 1.0
  ```

### 5. Parameter Parsing
- Added new parameters to `main.py` for configuring clustering constraints

## Usage

Follow the original LLM-ESR usage instructions with the addition of clustering-specific parameters. The clustering process is executed through the provided Jupyter notebook before running the main experiments.

## Reference

If you use this extension, please cite the original LLM-ESR work along with our paper (to be added).

## Poster
<img width="936" height="1322" alt="image" src="https://github.com/user-attachments/assets/9da966dc-80da-4eb3-9a98-a8e7fbb82f89" />

