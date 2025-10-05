# Surgical Capacity and Waiting Times for Planned Surgeries in Denmark  
*A Regional and Temporal Analysis Based in 2020-2025* 

# Problem Statement
**How does the capacity of _surgical_ hospital beds (available/normed) and surgical activity affect waiting times for planned surgery in the five Danish regions?**

---

## Sub-questions

### Development over time (surgery)
- How has the number of **surgically available** and **surgically normed** beds changed over time per region?
- Are there **seasonal fluctuations** in capacity, waiting-time buckets, and number of surgeries?

### Comparison of regions (surgery)
- Which regions have the highest/lowest **surgical capacity level**, and how has the development been?
- Which regions have **shorter waiting times relative to capacity** (e.g., waiting list per 100 surgically available beds)?

### Relationship between waiting time and capacity/activity (surgery)
- Is there a **correlation** between **surgically available beds** and the number of patients in waiting-time buckets (0–30, 31–60, 61–90, 90+ days)?
- Is **more surgically available beds** and/or **higher surgical activity** associated with **shorter waiting times**?

### Statistical analyses / models (surgery)
- Can a **(multi)linear regression** predict the waiting list / 90+ share based on **surgical capacity** and **surgical activity** (with month/region controls)?
- Can a **classification model** identify months with **high load** (e.g., top-25% 90+ waiting time)?  
  (Evaluated with **confusion matrix**, accuracy/F1.)
- Do **correlation heatmaps, tree models, clusters,** and **3D visualizations** provide consistent patterns?

- ## Run the Streamlit App

Make sure you are in the project root directory, then run:

```bash
streamlit run src/app.py
