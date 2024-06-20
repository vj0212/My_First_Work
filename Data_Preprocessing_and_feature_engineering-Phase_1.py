import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import norm, chi2_contingency
from numpy.linalg import svd

app_train = pd.read_csv(r"C:\Users\jhavi\Desktop\Python\CSI\CSI_Week_4\StudentId(CT_CSI_DS_1073)Assignment_4\application_train.csv")
titanic = pd.read_csv(r"C:\Users\jhavi\Desktop\Python\CSI\CSI_Week_4\StudentId(CT_CSI_DS_1073)Assignment_4\titanic.csv")

print("Application Train Dataset:")
print(app_train.info())

print("\nTitanic Dataset:")
print(titanic.info())

print("First few rows of Application Train Dataset:")
print(app_train.head())

print("\nFirst few rows of Titanic Dataset:")
print(titanic.head())

print("Summary statistics of Application Train Dataset:")
print(app_train.describe())

print("\nSummary statistics of Titanic Dataset:")
print(titanic.describe())

#Handling Missing Values
app_train_numeric = app_train.select_dtypes(include=[np.number])
app_train_categorical = app_train.select_dtypes(exclude=[np.number])

titanic_numeric = titanic.select_dtypes(include=[np.number])
titanic_categorical = titanic.select_dtypes(exclude=[np.number])

print("Numeric columns in Application Train Dataset:")
print(app_train_numeric.columns)

print("\nCategorical columns in Application Train Dataset:")
print(app_train_categorical.columns)

print("\nNumeric columns in Titanic Dataset:")
print(titanic_numeric.columns)

print("\nCategorical columns in Titanic Dataset:")
print(titanic_categorical.columns)

#Impute missing values for numeric columns using median strategy
imputer_numeric = SimpleImputer(strategy='median')
app_train_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(app_train_numeric), columns=app_train_numeric.columns)
titanic_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(titanic_numeric), columns=titanic_numeric.columns)

#Impute missing values for categorical columns using most frequent strategy
imputer_categorical = SimpleImputer(strategy='most_frequent')
app_train_categorical_imputed = pd.DataFrame(imputer_categorical.fit_transform(app_train_categorical), columns=app_train_categorical.columns)
titanic_categorical_imputed = pd.DataFrame(imputer_categorical.fit_transform(titanic_categorical), columns=titanic_categorical.columns)

#Combine numeric and categorical columns back together
app_train_imputed = pd.concat([app_train_numeric_imputed, app_train_categorical_imputed], axis=1)
titanic_imputed = pd.concat([titanic_numeric_imputed, titanic_categorical_imputed], axis=1)

print("Missing values after imputation in Application Train Dataset:")
print(app_train_imputed.isnull().sum())
print("\nMissing values after imputation in Titanic Dataset:")
print(titanic_imputed.isnull().sum())

#Encoding categorical variables
app_train_encoded = pd.get_dummies(app_train_imputed)
titanic_encoded = pd.get_dummies(titanic_imputed, drop_first=True)

print("Encoded Application Train Dataset shape:", app_train_encoded.shape)
print("Encoded Titanic Dataset shape:", titanic_encoded.shape)

#Standardization of Data
scaler_app_train = StandardScaler()
app_train_scaled = pd.DataFrame(scaler_app_train.fit_transform(app_train_encoded), columns=app_train_encoded.columns)
scaler_titanic = StandardScaler()
titanic_scaled = pd.DataFrame(scaler_titanic.fit_transform(titanic_encoded), columns=titanic_encoded.columns)

print("First few rows of the standardized Application Train Dataset:")
print(app_train_scaled.head())
print("\nFirst few rows of the standardized Titanic Dataset:")
print(titanic_scaled.head())

#Boxplot for Outlier Detection (Standardized Data)
plt.figure(figsize=(15, 8))
sns.boxplot(data=app_train_scaled.iloc[:, :10])  # Adjust the columns to visualize
plt.title("Boxplot for Application Train Dataset (Standardized)")
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=titanic_scaled)
plt.title("Boxplot for Titanic Dataset (Standardized)")
plt.xticks(rotation=90)
plt.show()

#Local Outlier Factor (LOF) (Standardized Data)
lof = LocalOutlierFactor(n_neighbors=20)
app_train_scaled['LOF'] = lof.fit_predict(app_train_scaled)

lof = LocalOutlierFactor(n_neighbors=20)
titanic_scaled['LOF'] = lof.fit_predict(titanic_scaled)

print("Outliers detected in Application Train Dataset (Standardized):")
print(app_train_scaled['LOF'].value_counts())
print("\nOutliers detected in Titanic Dataset (Standardized):")
print(titanic_scaled['LOF'].value_counts())

#Remove outliers detected by LOF
app_train_scaled_no_outliers = app_train_scaled[app_train_scaled['LOF'] == 1].drop(columns=['LOF'])
titanic_scaled_no_outliers = titanic_scaled[titanic_scaled['LOF'] == 1].drop(columns=['LOF'])

#Boxplot after removing outliers
plt.figure(figsize=(15, 8))
sns.boxplot(data=app_train_scaled_no_outliers.iloc[:, :10])  # Adjust the columns to visualize
plt.title("Boxplot for Application Train Dataset (Standardized, No Outliers)")
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=titanic_scaled_no_outliers)
plt.title("Boxplot for Titanic Dataset (Standardized, No Outliers)")
plt.xticks(rotation=90)
plt.show()

#Singular Value Decomposition
titanic_matrix_std = titanic_scaled_no_outliers.iloc[:, :5].to_numpy()
U_std, s_std, V_std = np.linalg.svd(titanic_matrix_std, full_matrices=False)
print("Singular values (Standardized):\n", s_std)

#Probability Distributions(Standardized Data)
#Example:Age distribution in Titanic dataset
sns.histplot(titanic_scaled_no_outliers['Age'], kde=True, stat="density", linewidth=0)
plt.title("Age Distribution in Titanic Dataset (Standardized, No Outliers)")
plt.show()

#Fit a normal distribution and overlay the plot
(mu_std, sigma_std) = norm.fit(titanic_scaled_no_outliers['Age'].dropna())
plt.hist(titanic_scaled_no_outliers['Age'].dropna(), bins=30, density=True, alpha=0.6, color='g')
xmin_std, xmax_std = plt.xlim()
x_std = np.linspace(xmin_std, xmax_std, 100)
p_std = norm.pdf(x_std, mu_std, sigma_std)
plt.plot(x_std, p_std, 'k', linewidth=2)
plt.title(f"Fit results (Standardized, No Outliers): mu = {mu_std:.2f},  sigma = {sigma_std:.2f}")
plt.show()

#Hypothesis Testing (Standardized Data)
#Example: Hypothesis testing on survival based on gender in Titanic dataset
contingency_table_std = pd.crosstab(titanic_scaled_no_outliers['Sex_male'], titanic_scaled_no_outliers['Survived'])
chi2_std, p_std, dof_std, expected_std = chi2_contingency(contingency_table_std)

print("Chi-square test (Standardized, No Outliers):")
print("Chi2:", chi2_std)
print("p-value:", p_std)
print("Degrees of freedom:", dof_std)
print("Expected frequencies:\n", expected_std)

#Additional Comparisons and Analysis
#Comparison of mean age between the two datasets
mean_age_app_train = app_train_scaled_no_outliers['DAYS_BIRTH'].mean() / -365  # Convert to positive age
mean_age_titanic = titanic_scaled_no_outliers['Age'].mean()
print(f"Mean age in Application Train Dataset: {mean_age_app_train:.2f} years")
print(f"Mean age in Titanic Dataset: {mean_age_titanic:.2f} years")

#Comparison of survival rates in Titanic dataset
survival_rate = titanic_scaled_no_outliers['Survived'].mean()
print(f"Survival rate in Titanic Dataset: {survival_rate:.2%}")

#Correlation analysis
correlation_app_train = app_train_scaled_no_outliers.corr()
correlation_titanic = titanic_scaled_no_outliers.corr()
print("Top correlations in Application Train Dataset:")
print(correlation_app_train.unstack().sort_values(kind="quicksort", ascending=False).drop_duplicates().head(10))
print("\nTop correlations in Titanic Dataset:")
print(correlation_titanic.unstack().sort_values(kind="quicksort", ascending=False).drop_duplicates().head(10))
