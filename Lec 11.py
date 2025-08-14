import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# ---------------- Q1: Mean, Std Dev, CV for Class One, Three, Five ----------------
class_one = np.array([85, 90, 88, 75, 78])
class_three = np.array([92, 95])
class_five = np.array([92, 95])  # Adjust if actual data differs

mean_one, std_one, cv_one = np.mean(class_one), np.std(class_one, ddof=1), np.std(class_one, ddof=1) / np.mean(class_one)
mean_three, std_three, cv_three = np.mean(class_three), np.std(class_three, ddof=1), np.std(class_three, ddof=1) / np.mean(class_three)
mean_five, std_five, cv_five = np.mean(class_five), np.std(class_five, ddof=1), np.std(class_five, ddof=1) / np.mean(class_five)

print("Q1: Class Performance")
print("Class One   - Mean:", mean_one, "Std Dev:", std_one, "CV:", cv_one)
print("Class Three - Mean:", mean_three, "Std Dev:", std_three, "CV:", cv_three)
print("Class Five  - Mean:", mean_five, "Std Dev:", std_five, "CV:", cv_five)

# ---------------- Q2: Average & Std Dev per Section in Classes ----------------
data_sections = {
    "Class": ["A","A","A","B","B","B","C","C","C","C","C"],
    "Section": ["S1","S2","S3","S4","S5","S7","S1","S2","S3","S4","S5"],
    "Score": [85,90,88,92,95,78,75,85,90,88,88]
}
df_sections = pd.DataFrame(data_sections)

avg_section = df_sections.groupby(["Class","Section"])["Score"].mean()
std_section = df_sections.groupby(["Class","Section"])["Score"].std(ddof=1)

print("\nQ2: Section Performance")
print("Average score per Section:\n", avg_section)
print("Standard Deviation per Section:\n", std_section)

# ---------------- Q3: Relationship Between Gender, Class & Pass/Fail ----------------
df_pass = pd.DataFrame({
    "Gender": ["Male","Female","Female","Male","Male","Female","Male","Female","Female","Male"],
    "Class": ["A","A","B","B","A","B","A","A","B","B"],
    "Passed": ["Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes"]
})

pass_table = pd.crosstab([df_pass["Gender"], df_pass["Class"]], df_pass["Passed"])
chi2, p, dof, expected = chi2_contingency(pass_table)

print("\nQ3: Gender & Class vs Pass/Fail")
print("Pass/Fail Crosstab:\n", pass_table)
print("Chi-square test: Chi2 =", chi2, "p-value =", p)
