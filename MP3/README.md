# Mini Project 3: Machine Learning for Analysis and Prediction

## Objective
This project applies regression, classification, and clustering algorithms to the IBM HR Analytics Attrition Dataset. The purpose is to analyze determinants of employee attrition, predict compensation levels, and identify clusters of employees with similar characteristics.

## Dataset and Exploration
The dataset includes demographic, job-related, and compensation attributes. Exploratory analysis indicated:

- **Income-related features** (MonthlyIncome, JobLevel, TotalWorkingYears) are strongly correlated.  
- **Tenure-related variables** (YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager) are moderately interrelated.  
- **Satisfaction measures** exhibit weak linear correlations with other numeric features, indicating their potential as independent drivers of attrition.

## Key Insights

1. **Primary factors driving attrition**  
   Ordered by importance:  
   1. Overtime  
   2. MonthlyIncome  
   3. TotalWorkingYears  
   4. Age  
   5. Satisfaction levels  
   6. YearsAtCompany  
   7. MonthlyRate  
   8. DistanceFromHome  
   9. DailyRate  
   10. HourlyRate  

   Employees working overtime with lower salaries and limited career progression are most at risk.

2. **Departmental and positional risk**  
   Employees in lower job levels, earning below the median, and with limited promotion history show the greatest attrition tendency.

3. **Compensation equity**  
   Gender pay disparities are not evident in the correlation analysis; income is explained primarily by job level and tenure.

4. **Work-life balance**  
   Distance from home and marital status influence perceived work-life balance, though the effect size is moderate.

5. **Education and satisfaction**  
   Education shows negligible correlation with job satisfaction, suggesting that higher education does not directly translate into greater workplace contentment.

6. **Clustering**  
   KMeans clustering yielded compact groupings concentrated at the lower end of income and tenure distributions, with no substantial outliers detected. The silhouette score indicated reasonable separation but suggested the workforce is largely homogeneous in certain dimensions.

## Challenges and Opportunities
- The attrition label is imbalanced, requiring careful model selection.  
- Strong correlations between compensation and tenure complicate feature independence assumptions.  
- Improvements may be achieved through advanced feature engineering and hyperparameter optimization of tree-based classifiers.
