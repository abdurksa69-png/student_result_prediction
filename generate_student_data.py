import pandas as pd
import numpy as np
import os
# Set random seed for reproducibility
np.random.seed(42)
# Number of students
n = 500
# Generate component marks
# Assignment: 10, Attendance: 10, Mid term: 20, Class test: 10, Semester final: 50
# Total = 100
assignment = np.random.randint(0, 11, n)        # 0-10
attendance = np.random.randint(0, 11, n)        # 0-10
mid_term = np.random.randint(0, 21, n)          # 0-20
class_test = np.random.randint(0, 11, n)        # 0-10
semester_final = np.random.randint(0, 51, n)   # 0-50
# Final marks = sum of all components (out of 100)
final_marks = assignment + attendance + mid_term + class_test + semester_final
# Pass / Fail (threshold: 40 out of 100)
result = ["Pass" if mark >= 40 else "Fail" for mark in final_marks]
# Create DataFrame
df = pd.DataFrame({
    "assignment": assignment,
    "attendance": attendance,
    "mid_term": mid_term,
    "class_test": class_test,
    "semester_final": semester_final,
    "final_marks": final_marks,
    "result": result
})
# Save file in the SAME folder as this script (works on Windows and Linux)
current_folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_folder, "student_data.csv")
try:
    df.to_csv(file_path, index=False)
    print("✅ Dataset saved successfully!")
    print("📁 Location:", file_path)
except Exception as e:
    print("❌ Error saving file:")
    print(e)
# Show first 5 rows
print("\nFirst 5 Rows:")
print(df.head())
# Show summary
print(f"\nPass: {result.count('Pass')} | Fail: {result.count('Fail')}")
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
