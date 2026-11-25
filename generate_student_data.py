
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# ------------------------------------------------------------------
# 0. Font setup (so any non-ASCII chars render cleanly)
# ------------------------------------------------------------------
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", context="notebook")

# ------------------------------------------------------------------
# 1. Generate the dataset (same logic as the original script)
# ------------------------------------------------------------------
np.random.seed(42)
n = 500

assignment     = np.random.randint(0, 11, n)   # 0-10
attendance     = np.random.randint(0, 11, n)   # 0-10
mid_term       = np.random.randint(0, 21, n)   # 0-20
class_test     = np.random.randint(0, 11, n)   # 0-10
semester_final = np.random.randint(0, 51, n)   # 0-50

final_marks = assignment + attendance + mid_term + class_test + semester_final
result = ["Pass" if mark >= 40 else "Fail" for mark in final_marks]

df = pd.DataFrame({
    "assignment": assignment,
    "attendance": attendance,
    "mid_term": mid_term,
    "class_test": class_test,
    "semester_final": semester_final,
    "final_marks": final_marks,
    "result": result,
})

# Save CSV
out_folder = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(out_folder, "student_data.csv")
df.to_csv(csv_path, index=False)
print("✅ Dataset saved successfully!")
print("📁 Location:", csv_path)

print(f"\nPass: {result.count('Pass')} | Fail: {result.count('Fail')}")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

# ------------------------------------------------------------------
# Output folder for plots
# By default plots save next to the CSV; change to a custom folder if you like
# ------------------------------------------------------------------
plot_folder = out_folder   # or: plot_folder = "/your/custom/path"
os.makedirs(plot_folder, exist_ok=True)

numeric_cols = ["assignment", "attendance", "mid_term",
                "class_test", "semester_final", "final_marks"]

# ==================================================================
# 2. Statistical summary: mean, median, std, skewness, kurtosis
# ==================================================================
summary = pd.DataFrame({
    "mean":      df[numeric_cols].mean(),
    "median":    df[numeric_cols].median(),
    "std":       df[numeric_cols].std(),
    "min":       df[numeric_cols].min(),
    "max":       df[numeric_cols].max(),
    "skewness":  df[numeric_cols].skew(),
    "kurtosis":  df[numeric_cols].kurt(),
})
summary_path = os.path.join(plot_folder, "distribution_statistics.csv")
summary.to_csv(summary_path)
print("\n=========== Distribution Statistics ===========")
print(summary.round(3).to_string())
print(f"\nSaved -> {summary_path}")

# ==================================================================
# 3. Histograms with KDE for each numeric column (2x3 grid)
# ==================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for ax, col in zip(axes, numeric_cols):
    sns.histplot(df[col], kde=True, color="#4C72B0",
                 edgecolor="white", alpha=0.85, ax=ax, bins=15)
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val,   color="red",   linestyle="--", linewidth=1.5,
               label=f"Mean={mean_val:.1f}")
    ax.axvline(median_val, color="green", linestyle="-",  linewidth=1.5,
               label=f"Median={median_val:.1f}")
    ax.set_title(f"Distribution of {col.replace('_', ' ').title()}", fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)

plt.suptitle("Histogram + KDE of All Components and Final Marks",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
hist_path = os.path.join(plot_folder, "01_histograms_kde.png")
plt.savefig(hist_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {hist_path}")

# ==================================================================
# 4. Box plots of all components side by side
# ==================================================================
fig, ax = plt.subplots(figsize=(11, 6))
box_data = df[["assignment", "attendance", "mid_term",
               "class_test", "semester_final", "final_marks"]]
sns.boxplot(data=box_data, palette="Set2", ax=ax)
ax.set_title("Box Plot of All Components and Final Marks", fontweight="bold")
ax.set_ylabel("Marks")
ax.set_xlabel("Component")
plt.tight_layout()
box_path = os.path.join(plot_folder, "02_boxplot_all_components.png")
plt.savefig(box_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved -> {box_path}")

# ==================================================================
# 5. Final marks distribution by result (overlaid Pass/Fail histogram)
# ==================================================================
fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(df[df["result"] == "Pass"]["final_marks"],
             color="#66C2A5", label="Pass", kde=True, alpha=0.6, ax=ax)
sns.histplot(df[df["result"] == "Fail"]["final_marks"],
             color="#FC8D62", label="Fail", kde=True, alpha=0.6, ax=ax)
ax.axvline(40, color="black", linestyle="--", linewidth=1.5,
           label="Pass threshold (40)")
ax.set_title("Final Marks Distribution by Result (Pass vs Fail)",
             fontweight="bold")
ax.set_xlabel("Final Marks")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
byresult_path = os.path.join(plot_folder, "03_final_marks_by_result.png")
plt.savefig(byresult_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved -> {byresult_path}")

# ==================================================================
# 6. Pass/Fail bar chart
# ==================================================================
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["result"].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=["#FC8D62" if r == "Fail" else "#66C2A5" for r in counts.index],
              edgecolor="black")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 3,
            str(val), ha="center", va="bottom", fontweight="bold")
ax.set_title("Pass vs Fail Counts", fontweight="bold")
ax.set_ylabel("Number of Students")
ax.set_xlabel("Result")
plt.tight_layout()
bar_path = os.path.join(plot_folder, "04_pass_fail_bar.png")
plt.savefig(bar_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved -> {bar_path}")

# ==================================================================
# 8. Combined distribution dashboard (all-in-one)
# ==================================================================
fig = plt.figure(figsize=(16, 12))

# Row 1: histogram + KDE for each of the 5 components
for i, col in enumerate(["assignment", "attendance", "mid_term",
                         "class_test", "semester_final"]):
    ax = fig.add_subplot(3, 4, i + 1)
    sns.histplot(df[col], kde=True, color="#55A868", bins=12, ax=ax)
    ax.set_title(f"{col.replace('_', ' ').title()}", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")

# Row 2: Pass/Fail bar, final marks histogram, box plot, by-result histogram
ax = fig.add_subplot(3, 4, 5)
counts = df["result"].value_counts()
ax.bar(counts.index, counts.values,
       color=["#FC8D62" if r == "Fail" else "#66C2A5" for r in counts.index],
       edgecolor="black")
for i, v in enumerate(counts.values):
    ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
ax.set_title("Pass / Fail", fontweight="bold")

ax = fig.add_subplot(3, 4, 6)
sns.histplot(df["final_marks"], kde=True, color="#C44E52", bins=20, ax=ax)
ax.set_title("Final Marks", fontweight="bold")

ax = fig.add_subplot(3, 4, 7)
sns.boxplot(data=df[numeric_cols], palette="Set2", ax=ax)
ax.set_title("Box Plot (All)", fontweight="bold")
ax.tick_params(axis='x', rotation=45)

ax = fig.add_subplot(3, 4, 8)
sns.histplot(df[df["result"] == "Pass"]["final_marks"],
             color="#66C2A5", label="Pass", kde=True, alpha=0.6, ax=ax)
sns.histplot(df[df["result"] == "Fail"]["final_marks"],
             color="#FC8D62", label="Fail", kde=True, alpha=0.6, ax=ax)
ax.axvline(40, color="black", linestyle="--", linewidth=1.5)
ax.set_title("Final Marks by Result", fontweight="bold")
ax.legend(fontsize=8)

plt.suptitle("Student Marks - Complete Distribution Dashboard",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
dash_path = os.path.join(plot_folder, "06_distribution_dashboard.png")
plt.savefig(dash_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved -> {dash_path}")

# ==================================================================
# 9. Final summary
# ==================================================================
print("\n=========== ALL DISTRIBUTION FEATURES GENERATED ===========")
print("1. Histograms + KDE             -> 01_histograms_kde.png")
print("2. Box plots (all components)   -> 02_boxplot_all_components.png")
print("3. Final marks by result        -> 03_final_marks_by_result.png")
print("4. Pass/Fail bar chart          -> 04_pass_fail_bar.png")
print("5. Combined dashboard           -> 06_distribution_dashboard.png")
print("6. Statistics CSV               -> distribution_statistics.csv")
print("\n✅ Done.")
