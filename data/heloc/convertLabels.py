import pandas as pd

# Load the HELOC dataset
df = pd.read_csv('original.csv')

# Convert "Bad" to 1 (Minority/Risk) and "Good" to 0 (Majority/Safe)
df['RiskPerformance'] = df['RiskPerformance'].map({'Bad': 1, 'Good': 0})

# Save it back out, overwriting the old file
df.to_csv('original.csv', index=False)
print("Conversion complete!")
exit()