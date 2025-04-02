---
title : Principal Component Analysis (PCA) in Machine Learning
date : 2025-03-25 14:00:00 +0800
categories : ["Machine Learning"]
tags :  ["Machine Learning", "Dimensionality Reduction"]
---

## Introduction
Principal Component Analysis (PCA) is a widely used dimensionality reduction technique in machine learning. It helps in transforming high-dimensional data into a lower-dimensional space while preserving as much variance as possible. This blog explores why, how, and when to use PCA, along with its mathematical justification and limitations.

---

## Why Use PCA?
- **Curse of Dimensionality**: High-dimensional data can lead to sparsity, making learning models inefficient.
- **Noise Reduction**: PCA removes less important features, improving generalization.
- **Visualization**: PCA helps in reducing dimensions to 2D or 3D for better visualization.
- **Computational Efficiency**: Reducing feature dimensions speeds up training and inference.

---

## How PCA Works: Step-by-Step
1. **Standardize the Data**  
   - Mean-center the data and scale it to have unit variance.

2. **Compute the Covariance Matrix**  
   - This captures relationships between variables.

3. **Compute Eigenvalues and Eigenvectors**  
   - Eigenvectors determine the principal components.
   - Eigenvalues indicate the amount of variance captured by each principal component.

4. **Sort and Select Top k Components**  
   - Choose the top k eigenvectors corresponding to the largest eigenvalues.

5. **Transform the Data**  
   - Project the original data onto the new subspace.

---

## Mathematical Justification
PCA finds a new set of basis vectors (principal components) such that:
- The new axes are **orthogonal**.
- The first principal component captures the maximum variance.
- Each subsequent component captures the next highest variance.

### 1. Maximizing Variance
Let \( X \) be our dataset with zero mean (after standardization), where each row represents a data point, and each column represents a feature.

We seek a unit vector \( v \) such that the **projected variance** is maximized. The projection of \( X \) onto \( v \) is:

$$
z = X v
$$

The variance of the projected data is:

$$
\text{Var}(z) = \frac{1}{n} \sum (X v)^T (X v) = v^T \left( \frac{1}{n} X^T X \right) v
$$

Since the **sample covariance matrix** is:

$$
C = \frac{1}{n} X^T X
$$

we rewrite variance as:

$$
\text{Var}(z) = v^T C v
$$

### 2. Finding the Principal Components
To maximize \( v^T C v \), we impose a constraint that \( v \) is a unit vector:

$$
||v||^2 = v^T v = 1
$$

Using **Lagrange multipliers**, we define:

$$
\mathcal{L} = v^T C v - \lambda (v^T v - 1)
$$

Differentiating and setting to zero:

$$
C v = \lambda v
$$

which is the **eigenvalue equation**. This means:
- The eigenvectors of \( C \) define the new principal directions.
- The eigenvalues \( \lambda \) represent the amount of variance captured by each eigenvector.

### 3. Ordering Eigenvectors by Variance
Since larger eigenvalues correspond to higher variance, we:
1. Compute eigenvalues \( \lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_d \).
2. Select the top \( k \) eigenvectors to form a new basis.
3. Transform the data:  

   $$
   X' = X V_k
   $$

where \( V_k \) contains the top \( k \) eigenvectors.

Thus, PCA **rotates the coordinate system** to align with the directions of maximum variance, reducing dimensionality while preserving important information.

---

## When to Use PCA?
- **High-dimensional data where features are correlated**
- **Reducing overfitting in models**
- **Visualizing complex datasets in lower dimensions**
- **Speeding up machine learning algorithms**

---

## When PCA Doesn't Work Well
- **When feature importance is needed**: PCA transforms features, making them less interpretable.
- **For non-linear data**: PCA assumes a linear relationship, failing for non-linearly separable data.
- **When all features are equally important**: PCA removes variance-based information, potentially discarding useful features.
- **When feature scaling is inconsistent**: PCA is sensitive to scale; improper scaling can lead to misleading results.

---

## Best Practices for Using PCA
- **Always standardize data before applying PCA** to prevent features with larger magnitudes from dominating.
- **Choose the right number of components** by analyzing the explained variance ratio.
- **Use PCA only when dimensionality reduction is needed**—not as a default preprocessing step.
- **Consider Kernel PCA** for non-linear relationships.

---

## Conclusion
PCA is a powerful technique for dimensionality reduction, helping in visualization, noise reduction, and improving model efficiency. However, it has limitations and should be used appropriately based on the dataset and problem requirements.

---

<script type="text/javascript" async
  src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" async
  id="MathJax-script"
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
