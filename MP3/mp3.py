import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm 
import graphviz
from sklearn.metrics import r2_score
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --- Load ---
def load_employee(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, sep=",")
    else:
        df = pd.read_excel(path)
        if any(str(c).lower().startswith("unnamed") for c in df.columns):
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
        return df

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_DIR = os.path.join(BASE_DIR, "data")             

employee_df = load_employee(os.path.join(DATA_DIR, "employee_attrition.csv"))

# --- Clean ---
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize column names (snake_case)
    df.columns = [str(c).strip().replace(" ", "").replace("-", "") for c in df.columns]
    
    # keep relevant columns
    keep_cols = [
        "Age","Attrition","BusinessTravel","DailyRate","Department","DistanceFromHome",
        "Education","EducationField","EmployeeCount","EmployeeNumber","EnvironmentSatisfaction","Gender","HourlyRate",
        "JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome","MonthlyRate",
        "NumCompaniesWorked","Over18","OverTime","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction",
        "StandardHours","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance",
        "YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    
    # drop duplicates & NA
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df

employee_clean_df = clean(employee_df)
employee_clean_df.info()
# print(employee_clean_df.sample(5))

# --- README ---
print(employee_clean_df["Over18"].value_counts())

drop_cols = [c for c in["Over18"] if c in employee_clean_df.columns]
employee_clean_df = employee_clean_df.drop(columns=drop_cols)

# --- Histograms ---
def plot_employee_histogram(employee_clean_df):
    employee_clean_df.hist(bins=30, figsize=(20, 12))
    plt.show()

plot_employee_histogram(employee_clean_df)

def plot_correlation_heatmap(df, title="Correlation Heatmap"):

    plt.figure(figsize=(20,12))
    corr = df.select_dtypes(include=[np.number]).corr()
    sb.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.show()

plot_correlation_heatmap(employee_clean_df, title="Employee Data Correlation Heatmap")
# Heatmap shows a strong correlation between MonthlyIncome and TotalWorkingYears
# Heatmap shows redundant columns: EmployeeCount, StandardHours
# Heatmap shows a strong correlation between JobLevel and MonthlyIncome
# Heatmap shows a strong correlation between JobLevel and TotalWorkingYears
# Heatmap shows a strong correlation between YearsAtCompany and TotalWorkingYears
# Heatmap shows a strong correlation between YearsAtCompany and YearsWithCurrManager
# Heatmap shows a strong correlation between YearsInCurrentRole and YearsWithCurrManager
# Heatmap shows a strong correlation between YearsAtCompany and YearsInCurrentRole
# Heatmap shows a strong correlation between TotalWorkingYears and MonthlyIncome

# --- Scatterplots ---
def plot_feature_scatter(df, x_col, y_col, hue=None, palette="Set1", title=None, alpha=0.6):
  
    plt.figure(figsize=(12,6))
    sb.scatterplot(data=df, x=x_col, y=y_col, hue=hue, palette=palette, alpha=alpha)
    if title is None:
        title = f"{y_col.capitalize()} vs {x_col.capitalize()}"
    plt.title(title)
    plt.show()

"""
plot_feature_scatter(employee_clean_df, x_col="TotalWorkingYears", y_col="MonthlyIncome", hue="Attrition", title="Monthly Income vs Total Working Years")
plot_feature_scatter(employee_clean_df, x_col="YearsAtCompany", y_col="TotalWorkingYears", hue="Attrition", title="Monthly Income vs Years at Company")
plot_feature_scatter(employee_clean_df, x_col="YearsInCurrentRole", y_col="YearsWithCurrManager", hue="Attrition", title="Years with Current Manager vs Years in Current Role")
plot_feature_scatter(employee_clean_df, x_col="YearsAtCompany", y_col="YearsWithCurrManager", hue="Attrition", title="Years with Current Manager vs Years at Company")

plot_feature_scatter(employee_clean_df, x_col="JobLevel", y_col="MonthlyIncome", hue="Attrition", title="Monthly Income vs Job Level")
plot_feature_scatter(employee_clean_df, x_col="JobLevel", y_col="TotalWorkingYears", hue="Attrition", title="Total Working Years vs Job Level")
plot_feature_scatter(employee_clean_df, x_col="JobLevel", y_col="YearsAtCompany", hue="Attrition", title="Years at Company vs Job Level")
plot_feature_scatter(employee_clean_df, x_col="JobLevel", y_col="YearsInCurrentRole", hue="Attrition", title="Years in Current Role vs Job Level") 

"""

def plot_employee_boxplots(df):
    df.plot(kind='box', subplots=True, layout=(3,12), sharex=False, sharey=False,figsize=(24,12))
    plt.show()

plot_employee_boxplots(employee_clean_df)   

# --- Engineering ---
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = df.copy()
    
    # Binary encode
    df_eng["Attrition"] = df_eng["Attrition"].map({"Yes": 1, "No": 0})
    if "OverTime" in df_eng.columns:
        df_eng["OverTime"] = df_eng["OverTime"].map({"Yes": 1, "No": 0})

    # Tenure buckets
    if "YearsAtCompany" in df_eng.columns:
        df_eng["TenureBucket"] = pd.cut(
            df_eng["YearsAtCompany"], 
            bins=[-1,3,6,10,40], 
            labels=["0-3","4-6","7-10","10+"]
        )

    # Income bands (5 bins)
    if "MonthlyIncome" in df_eng.columns:
        df_eng["IncomeBand"] = pd.qcut(df_eng["MonthlyIncome"], 5, labels=False)

    # Average satisfaction score
    sats = ["EnvironmentSatisfaction","JobSatisfaction","RelationshipSatisfaction"]
    if all(c in df_eng.columns for c in sats):
        df_eng["AvgSatisfaction"] = df_eng[sats].mean(axis=1)

    # One-hot encode categorical vars
    cat_cols = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus"]
    cat_cols = [c for c in cat_cols if c in df_eng.columns]
    df_eng = pd.get_dummies(df_eng, columns=cat_cols, drop_first=True)
    
    return df_eng

employee_engineered_df = engineer_features(employee_clean_df)
print(employee_engineered_df.head())

# --- Remove Outliers ---
def remove_all_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean

employee_clean_df = remove_all_outliers(employee_engineered_df)

# --- Train a model: Multiple linear regression ---
df_model = employee_engineered_df.copy()

df_model = pd.get_dummies(df_model, drop_first=True)

X = df_model.drop(columns=["MonthlyIncome"])

X.head()

y = df_model["MonthlyIncome"]

y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

linreg = LinearRegression()
linreg.fit(X_train,y_train)

# --- Association ---
print('b0 =', linreg.intercept_)
coef_table = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": linreg.coef_
})
# Question 3. in README.md
print(coef_table.sort_values(by="coefficient", ascending=False).head(15))  # Top 15 features
print(coef_table.sort_values(by="coefficient", ascending=True).head(15))   # Bottom 15 features

# --- Prediction ---
y_predicted = linreg.predict(X_test)
print(y_predicted)

# calculate MAE using scikit-learn
print(sm.mean_absolute_error(y_test, y_predicted))

# calculate MSE using scikit-learn
print(sm.mean_squared_error(y_test, y_predicted))

# calculate RMSE using scikit-learn
print(np.sqrt(sm.mean_squared_error(y_test, y_predicted)))

# Explained variance (1 would be the best prediction)
eV = round(sm.explained_variance_score(y_test, y_predicted), 2)
print('Explained variance score ',eV )

# R-squared
print(r2_score(y_test, y_predicted))

plt.title('Multiple Linear Regression')
plt.scatter(y_test, y_predicted, color='blue')
plt.show()

# --- Train a model: Classification ---
df_cls = employee_engineered_df.copy()

df_cls = pd.get_dummies(df_cls, drop_first=True)
print(df_cls.info())

y = df_cls['Attrition']
X = df_cls.drop('Attrition', axis=1)

set_prop = 0.15

seed = 12

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=set_prop, random_state=seed)

params = {'max_depth': 5}
classifier = DecisionTreeClassifier(**params)
classifier2 = RandomForestClassifier(n_estimators = 100, max_depth = 6)

classifier.fit(X_train, y_train)
classifier2.fit(X_train, y_train)

importances = classifier2.feature_importances_
features = X.columns

# Question 1. in README.md
feat_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot top 10
plt.figure(figsize=(10,6))
plt.barh(feat_importance["Feature"][:10][::-1], 
         feat_importance["Importance"][:10][::-1], color="skyblue")
plt.xlabel("Importance")
plt.title("Top 10 Factors Influencing Attrition")
plt.show()
""""
# Export decision tree to PDF in the data folder
out_path = os.path.join(DATA_DIR, "attrition_tree")

dot_data = tree.export_graphviz(
    classifier,
    out_file=None,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data, format="pdf")
graph.render(out_path, cleanup=True)

print(f"Decision tree saved to: {out_path}.pdf")
"""
# --- Model validation ---

# Predict the labels of the test data
y_testp = classifier.predict(X_test)

print ("Accuracy is ", accuracy_score(y_test,y_testp))

confusion_mat = confusion_matrix(y_test,y_testp)
print(confusion_mat)

confusion = pd.crosstab(y_test,y_testp)
print(confusion)

# Confusion matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

ticks = np.arange(2)
plt.xticks(ticks, ["No", "Yes"])
plt.yticks(ticks, ["No", "Yes"])

plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.tight_layout()
plt.show()

# Heatmap
plt.figure(figsize=(6,4))
sb.heatmap(confusion_mat,
            annot=True,        # show the numbers
            fmt="d",           # integer formatting
            cmap="Blues",      # color map
            xticklabels=["No", "Yes"],  # predicted labels
            yticklabels=["No", "Yes"])  # true labels

plt.ylabel("True labels")
plt.xlabel("Predicted labels")
plt.title("Confusion Matrix")
plt.show()

# --- Clustering ---
df_cluster = employee_engineered_df.copy()

# Drop columns that don't make sense for clustering
drop_cols = [c for c in ["EmployeeNumber", "EmployeeCount", "StandardHours", "Attrition"] if c in df_cluster.columns]
df_cluster = df_cluster.drop(columns=drop_cols)

# Ensure we only cluster on numeric columns
df_cluster = df_cluster.select_dtypes(include=["number"])

# Scale
scaler = StandardScaler()
X_clust = scaler.fit_transform(df_cluster)

# Search k
ks = range(2, 11)
inertias = []
sil_scores = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clust)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_clust, labels))

# Pick best k (max silhouette)
best_k = ks[int(np.argmax(sil_scores))]
best_sil = sil_scores[int(np.argmax(sil_scores))]
print(f"Best k by silhouette: k={best_k}, silhouette={best_sil:.4f}")

# Fit final model and attach labels
best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_labels = best_kmeans.fit_predict(X_clust)
employee_engineered_df["Cluster"] = final_labels  # keep cluster on your main engineered DF

# --- Diagnostics: Elbow & Silhouette plots ---
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(list(ks), inertias, marker="o")
plt.title("Elbow (Inertia) vs k")
plt.xlabel("k"); plt.ylabel("Inertia")

plt.subplot(1,2,2)
plt.plot(list(ks), sil_scores, marker="o")
plt.title("Silhouette vs k")
plt.xlabel("k"); plt.ylabel("Silhouette score")
plt.tight_layout()
plt.show()