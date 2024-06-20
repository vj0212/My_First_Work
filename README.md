# My_First_Work
This repository contains Python code for preprocessing and analyzing two datasets: the Application Train dataset and the Titanic dataset.

### Data Preprocessing and Analysis in Python

#### Steps and Techniques Used:

1. **Data Loading and Exploration:**
   - Loaded datasets from CSV files into Pandas DataFrames.
   - Examined basic information and summary statistics of both datasets.

2. **Handling Missing Values:**
   - Separated numeric and categorical columns.
   - Imputed missing numeric values using median strategy.
   - Imputed missing categorical values using most frequent strategy.

3. **Encoding Categorical Variables:**
   - Applied one-hot encoding to categorical columns.

4. **Standardization:**
   - Standardized the data using `StandardScaler` to ensure each feature contributes equally.

5. **Outlier Detection and Removal:**
   - Detected outliers using Local Outlier Factor (LOF) on standardized data.
   - Removed outliers identified by LOF.

6. **Dimensionality Reduction:**
   - Applied Singular Value Decomposition (SVD) to reduce dimensionality and extract significant features.

7. **Probability Distributions:**
   - Visualized and fitted probability distributions, such as the normal distribution, to key variables.

8. **Hypothesis Testing:**
   - Conducted hypothesis testing (e.g., chi-square test) to analyze relationships between categorical variables.

9. **Additional Analysis:**
   - Compared mean ages and survival rates across datasets.
   - Explored correlations between features within each dataset.

#### Visualizations and Results:

- **Boxplots:** Visualized data distributions and outliers before and after preprocessing.
- **Histograms:** Examined distributions of key variables with fitted probability distributions.
- **Correlation Analysis:** Identified significant correlations within datasets.

#### Conclusion:

This repository provides a comprehensive example of data preprocessing techniques, exploratory analysis, and statistical testing using Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn. It serves as a useful reference for handling real-world datasets, ensuring data quality, and deriving meaningful insights through systematic analysis.

For further details and the complete code, please refer to the Python scripts in this repository.


