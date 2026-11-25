# similarity-search-high-dimensional-vectors

Naive linear scan, pivot-based and iDistance methods for k-nearest neighbour and range queries on 10-dimensional vectors, using Euclidean distance (L2).

Assignment 3, Data Management (Διαχείριση Σύνθετων Δεδομένων), University of Ioannina, 2025-26.
Team: Christina Danae Dafermou (5206), Christos Valeras (5182)

## Contents
- `assignment3.py`: loads the data, selects the pivots, builds the iDistance index and answers the queries with the three methods.
- `experiments.py`: runs the experiments for different values of eps and k, and creates the plots.
- `data10K10.txt`: 10,000 vectors with 10 dimensions each.
- `queries10.txt`: query vectors.
- `knn_comps.png`, `knn_time.png`, `range_comps.png`, `range_time.png`: experiment plots.
- `report_diaxeirisi_3_5206_5182.pdf`: project report.

## Requirements
- Python 3
- matplotlib (only for `experiments.py`)

## How to Run
1. Clone the repository: `git clone https://github.com/dafermoudanae/similarity-search-high-dimensional-vectors.git`
2. Run the program with the default values (5 pivots, eps = 0.2, k = 5):
   `python assignment3.py`
3. Or give the values yourself: `python assignment3.py <numpivots> <eps> <k> [data_file] [query_file]`
4. Run the experiments: `python experiments.py`
