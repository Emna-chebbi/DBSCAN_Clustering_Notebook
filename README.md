# DBSCAN Customer Segmentation

A machine learning project exploring **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** for customer segmentation.

The project combines theoretical explanations with practical experimentation, demonstrating how DBSCAN can discover clusters based on **data density**, handle **irregularly shaped clusters**, and identify **outliers/noise** without requiring the number of clusters in advance.

The algorithm is applied to the **Mall Customer Segmentation dataset**, using Annual Income and Spending Score to identify customer segments and compare the results with KMeans.

---

## 🛍️ Dataset

The project uses the **Mall Customer Segmentation dataset**, containing information about 200 customers.

The dataset includes:

| Feature                  | Description                  |
| ------------------------ | ---------------------------- |
| `CustomerID`             | Customer identifier          |
| `Gender`                 | Customer gender              |
| `Age`                    | Customer age                 |
| `Annual Income (k$)`     | Annual income                |
| `Spending Score (1-100)` | Mall-assigned spending score |

The target is not a prediction variable because this is an **unsupervised learning** problem.

Instead, the goal is to discover natural groups of customers.

---

## Exploratory Data Analysis

The EDA investigates:

* Dataset structure
* Missing values
* Duplicate records
* Summary statistics
* Feature distributions
* Income vs. Spending Score
* Age vs. Spending Score
* Correlations between numerical features

### Key Finding

The **Annual Income vs. Spending Score** relationship shows the clearest clustering structure.

The Age vs. Spending Score relationship is considerably more dispersed, making Income and Spending Score more suitable for the main DBSCAN analysis.

Therefore, the clustering model uses:

```text
Annual Income (k$)
Spending Score (1-100)
```

### Select Clustering Features

The model uses:

```python
feature_cols = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]
```

### Standardize Features

The features are standardized using `StandardScaler`.

This is important because DBSCAN relies on distances. Without scaling, differences in feature ranges could cause one variable to dominate the distance calculation.

---

## Choosing DBSCAN Hyperparameters

### `min_samples`

The project uses:

```text
min_samples = 5
```

This provides a reasonable starting point for a two-dimensional dataset.

### `eps`

The `eps` parameter is selected using a **k-distance graph**.

The distance to each point's 5th nearest neighbor is calculated and sorted. The resulting graph is inspected for an **elbow**, representing a transition between dense and sparse regions.

The selected value is:

```text
eps = 0.35
```

Because the features are standardized, `eps` is expressed in standardized distance units.

---

## DBSCAN Model

The final DBSCAN configuration is:

```python
DBSCAN(
    eps=0.35,
    min_samples=5
)
```

The model is trained on the standardized:

* Annual Income
* Spending Score

---

## DBSCAN Results

With the selected parameters, DBSCAN discovered:

* **6 customer clusters**
* **23 noise/outlier points**
* **200 total customers**
* **11.5% identified as noise**

DBSCAN discovered the number of clusters automatically from the density structure of the data.

The 23 noise points represent customers who do not clearly belong to any dense customer segment.

---

## Cluster Evaluation

The project uses the **Silhouette Score** to evaluate cluster separation.

The Silhouette Score ranges from:

```text
-1 → Poor clustering
 0 → Points near cluster boundaries
+1 → Well-separated clusters
```

### DBSCAN vs. KMeans

| Model        | Silhouette Score |
| ------------ | ---------------: |
| DBSCAN       |        **0.558** |
| KMeans (k=6) |            0.540 |

DBSCAN achieved a slightly higher Silhouette Score.


---

## Customer Segments

The resulting clusters can be interpreted using their average age, income, and spending score.

| Cluster   | Customer Profile                                      |
| --------- | ----------------------------------------------------- |
| Cluster 0 | Young customers with low income but high spending     |
| Cluster 1 | Older customers with low income and very low spending |
| Cluster 2 | Customers with low income and moderate spending       |
| Cluster 3 | Customers with average income and spending            |
| Cluster 4 | Higher-income customers with very high spending       |
| Cluster 5 | Higher-income customers with very low spending        |
