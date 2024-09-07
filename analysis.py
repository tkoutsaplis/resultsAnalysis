import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd 

# Function to group raw data into sublists of 4 elements
def clean_and_organize_data(raw_data):
    numbers = [int(num) for num in raw_data.split() if num.isdigit()]
    return [numbers[i:i+4] for i in range(0, len(numbers), 4)]

# Clean and organize the data
overhand_data =     clean_and_organize_data("4 3 4 5 5 3 4 5 4 2 5 4 4 4 5 4 4 4 4 5 5 2 5 5 2 4 3 5 2 2 4 5 4 4 5 5 5 4 5 5 4 4 3 4 4 3 5 5 4 4 5 5 5 4 5 5 4 2 4 4 4 3 5 5 5 2 4 5 3 2 4 2 4 3 4 3 5 2 5 5 3 2 4 5 5 4 5 5 5 5 5 5 5 2 4 5 5 2 5 5 4 4 5 5 4 4 5 5 4 3 5 5 5 4 3 5 4 3 4 5 5 4 4 5 5 4 5 5 4 4 4 4 5 4 4 3 5 2 4 5 4 4 4 5")
underhand_data =    clean_and_organize_data("4 3 4 5 5 4 5 5 5 2 3 4 5 2 4 4 5 4 5 5 5 5 5 5 4 2 4 4 3 3 4 5 4 4 4 5 5 4 4 4 5 5 5 5 5 5 4 5 5 5 5 5 5 4 4 5 5 4 4 5 4 2 4 5 4 4 4 5 4 4 5 5 5 5 4 5 4 4 5 5 5 4 5 5 4 3 5 4 4 3 5 4 4 2 3 4 5 3 4 5 4 2 4 5 4 1 3 4 5 3 4 5 2 4 5 5 4 2 4 4 4 3 4 5 4 4 4 3 5 3 4 4 5 5 5 5 4 2 4 4 5 2 4 5 5 3 2 5 4 3 5 4 5 4 3 5 4 5 4 5 4 4 4 5 5 4 4 5 5 4 5 5 4 2 4 4")
side_data =         clean_and_organize_data("5 4 3 5 4 5 4 4 5 5 5 5 4 4 5 5 5 4 5 5 4 2 2 1 5 5 4 5 4 4 3 5 5 5 5 5 4 4 3 5 5 4 5 5 4 3 4 3 5 4 5 5 4 3 5 4 5 5 5 5 4 2 4 3 4 4 4 5 4 3 5 5 4 5 5 4 4 4 4 5 5 2 4 5 4 4 4 4 4 4 5 4 5 4 4 5 4 3 5 5 3 4 5 5 4 3 5 4 5 3 4 5")

# Function to extract a specific metric from the data
def extract_metric(data, index):
    return [row[index] for row in data]

# Prepare data for each metric
metrics = ['Appropriateness', 'Naturalness', 'Predictability', 'Safety']
prepared_data = {}

for i, metric in enumerate(metrics):
    prepared_data[metric] = {
        'Overhand': extract_metric(overhand_data, i),
        'Underhand': extract_metric(underhand_data, i),
        'Side': extract_metric(side_data, i)
    }

# Print summary of prepared data by metric
print("\n")
print("----- Results grouped by metric: -----")
for metric, methods in prepared_data.items():
    print(f"\n{metric}:")
    for method, values in methods.items():
        print(f"  {method}: {len(values)} samples, Mean: {np.mean(values):.2f}, Std: {np.std(values):.2f}")

# Print summary of prepared data by handover method
print("\n----- Results grouped by handover method: -----")
methods = ['Overhand', 'Underhand', 'Side']
for method in methods:
    print(f"\n{method}:")
    for metric in metrics:
        values = prepared_data[metric][method]
        print(f"  {metric}: Mean: {np.mean(values):.2f}, Std: {np.std(values):.2f}")

# Perform t-tests
print("\n----- T-Test Results: -----")
for metric in metrics:
    print(f"\n{metric}:")
    for method1, method2 in [('Overhand', 'Underhand'), ('Overhand', 'Side'), ('Underhand', 'Side')]:
        t_stat, p_value = stats.ttest_ind(prepared_data[metric][method1], prepared_data[metric][method2])
        print(f"  {method1} vs {method2}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

# Calculate overall mean and std for each metric
overall_metrics = {}
for metric in metrics:
    all_values = prepared_data[metric]['Overhand'] + prepared_data[metric]['Underhand'] + prepared_data[metric]['Side']
    overall_metrics[metric] = {
        'Mean': np.mean(all_values),
        'Std': np.std(all_values)
    }

print("\n----- Overall Metrics: -----")
for metric, stats in overall_metrics.items():
    print(f"\n{metric}:")
    print(f"  Mean: {stats['Mean']:.2f}")
    print(f"  Std: {stats['Std']:.2f}")

# Perform three-way repeated measures ANOVA
data = []

# Create a list of dictionaries to be used in the pandas DataFrame
for metric, methods in prepared_data.items():
    for method, scores in methods.items():
        for i, score in enumerate(scores):
            data.append({
                'Participant': i + 1,
                'Method': method,
                'Metric': metric,
                'Score': score
            })

df = pd.DataFrame(data) # Create pandas DataFrame from list of dictionaries

model = ols('Score ~ C(Method) + C(Metric) + C(Method):C(Metric)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\n----- Three-Way Repeated Measures ANOVA Results: -----")
print(anova_table)
print("\n")