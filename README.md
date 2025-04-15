# Big Data Movie Recommendations using ALS (PySpark)

This project builds a movie recommendation system using the **Alternating Least Squares (ALS)** algorithm in **PySpark**. The focus is on reducing data sparsity, tuning hyperparameters, and improving validation accuracy using a large ratings dataset.

---

## 🧠 Project Highlights

- **Framework:** PySpark (MLlib)
- **Goal:** Predict user ratings for movies using collaborative filtering
- **Challenges Addressed:**
  - Large dataset (~163MB)
  - High sparsity of user-movie matrix
  - Performance optimization

---

## ⚙️ Methodology

1. **Data Loading & Schema Inference**
   - Loaded large CSV using PySpark's DataFrame API
   - Used both inferred and manual schemas for precision

2. **Sparsity Reduction**
   - Random sampling and systematic user sampling to reduce matrix sparsity
   - Reduced RMSE from **~3.89** to **~0.95** after sampling

3. **Model Training**
   - Trained ALS model using training/validation/test splits
   - Implemented grid search over:
     - `rank` (latent factors): 4, 8, 12  
     - `regularization` (lambda): 0.1, 0.2  
     - `iterations`: 15, 30  

4. **Evaluation Metrics**
   - RMSE (Root Mean Squared Error) calculated for validation and testing sets
   - Used PySpark RDD joins and transformations for error computation

---

## 📊 Results

| Model Stage                         | Validation RMSE |
|-------------------------------------|------------------|
| Initial model (raw data)            | ~3.89            |
| After reducing sparsity             | ~0.956           |
| After hyperparameter tuning         | **~0.817**       |

**Best Hyperparameters:**
- Rank: `12`
- Regularization: `0.1`
- Iterations: `30`

---

## 📂 Output

- Evaluation results saved as Spark RDD at: `./results`

---

## 💡 Key Learnings

- Importance of reducing matrix sparsity for recommendation systems
- Practical experience with distributed training and evaluation
- Efficient use of caching/persisting in PySpark

---

## 🚀 How to Run

> Requires PySpark installed and configured. Dataset paths must be adjusted based on local setup.

```bash
spark-submit recommendation_system.py
