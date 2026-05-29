# 🤖 Machine Learning Examples

A collection of machine learning algorithms implemented in Python and Jupyter Notebook, covering supervised, unsupervised, and dimensionality reduction techniques — with real-world datasets and worked examples.

---

## 📂 Repository Structure

```
Machine-learning-examples/
├── Clustering/               # K-Means and other clustering algorithms
├── Data/                     # Shared datasets (e.g. diabetes, bank marketing)
├── Decision Trees/           # Decision tree classifiers and regressors
├── Linear Regression/        # Linear regression with insurance data
├── Logistic Regression/      # Binary classification coursework (bank marketing)
├── Neural Network/           # Neural network implementations
├── PCA_Association_Rules/    # Dimensionality reduction & association rule mining
└── Support Vector Machine/   # SVM on Iris and other datasets
```

---

## 🧠 Topics Covered

| Folder | Algorithm | Type |
|---|---|---|
| `Clustering` | K-Means, Hierarchical | Unsupervised |
| `Decision Trees` | CART, ID3 | Supervised |
| `Linear Regression` | OLS, Ridge | Supervised |
| `Logistic Regression` | Binary Classification | Supervised |
| `Neural Network` | MLP, Feedforward | Supervised |
| `PCA_Association_Rules` | PCA, Apriori | Unsupervised / Dimensionality Reduction |
| `Support Vector Machine` | SVM (Iris dataset) | Supervised |

---

## 📊 Datasets Used

- **Bank Marketing** (`bank-full.csv`) — UCI repository, used in Logistic Regression
- **Diabetes** — used in Data / regression examples
- **Insurance** — used in Linear Regression
- **Iris** — used in SVM visualisation
- **UNdata_Export** - Used in agriculture trade value forecasting

> Datasets are stored in the `Data/` folder or downloaded from public sources (UCI, sklearn) within the notebooks.

---

## 🚀 Getting Started

### Prerequisites

Python 3.8+ is required. Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running a Notebook

```bash
git clone https://github.com/Dkishordev/Machine-learning-examples.git
cd Machine-learning-examples
jupyter notebook
```

Then navigate to any folder and open the `.ipynb` file of your choice.

---

## 📁 Module Summaries

### 🔵 Clustering
Unsupervised grouping of data points using K-Means and related methods. Includes elbow method for optimal `k` selection and cluster visualisation.

### 🌳 Decision Trees
Tree-based classifiers and regressors. Covers Gini impurity, entropy, pruning, and decision boundary plots.

### 📈 Linear Regression
Regression modelling using the insurance dataset. Explores feature relationships, residual analysis, and model performance (R², RMSE).

### 📉 Logistic Regression
Binary classification coursework predicting bank term deposit subscriptions. Full pipeline: EDA → preprocessing → model training → ROC curve → feature importance.

### 🧬 Neural Network
Feedforward neural network examples. Covers architecture design, activation functions, training loops, and loss curves.

### 🔻 PCA & Association Rules
- **PCA:** Dimensionality reduction and visualisation of high-dimensional data.
- **Association Rules:** Apriori algorithm for market basket analysis (support, confidence, lift).

### 🔷 Support Vector Machine
SVM classification on the Iris dataset with hyperplane and decision boundary visualisation. Covers linear and RBF kernels.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellowgreen?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-lightblue?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-blueviolet)



