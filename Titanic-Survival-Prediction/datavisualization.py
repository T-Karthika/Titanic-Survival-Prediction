import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("dataset/train.csv")

df['Sex'] = df['Sex'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')

plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.savefig(f"{output_dir}/missing_values.png")
plt.show()

sns.countplot(x='Survived', data=df, palette='coolwarm')
plt.xlabel("Survival (0 = No, 1 = Yes)")
plt.ylabel("Passenger Count")
plt.title("Survival Count")
plt.savefig(f"{output_dir}/survival_count.png")
plt.show()

sns.barplot(x='Sex', y='Survived', data=df, palette='Blues')
plt.xlabel("Gender (Male/Female)")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Gender")
plt.savefig(f"{output_dir}/survival_by_gender.png")
plt.show()

sns.histplot(df['Age'].dropna(), bins=30, kde=True, color='purple')
plt.xlabel("Age")
plt.ylabel("Passenger Count")
plt.title("Age Distribution of Passengers")
plt.savefig(f"{output_dir}/age_distribution.png")
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df, palette='magma')
plt.xlabel("Passenger Class (1st, 2nd, 3rd)")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Passenger Class")
plt.savefig(f"{output_dir}/survival_by_class.png")
plt.show()

sns.boxplot(x='Survived', y='Fare', data=df, palette='Set2')
plt.xlabel("Survival (0 = No, 1 = Yes)")
plt.ylabel("Fare Amount")
plt.title("Fare Distribution by Survival")
plt.savefig(f"{output_dir}/fare_distribution.png")
plt.show()


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.show()
