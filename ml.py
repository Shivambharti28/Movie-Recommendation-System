import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. LOAD & CLEAN DATA
# ==========================================
print("Loading dataset...")
df = pd.read_pickle("models/df.pkl")

# Keep relevant columns for ML
ml_df = df[['title', 'budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']].copy()

# Drop rows where title is missing
ml_df = ml_df.dropna(subset=['title'])

# Fill missing numerical values with mean or 0
ml_df['runtime'] = ml_df['runtime'].fillna(ml_df['runtime'].mean())
ml_df['vote_average'] = ml_df['vote_average'].fillna(ml_df['vote_average'].mean())
ml_df['vote_count'] = ml_df['vote_count'].fillna(0)
ml_df['budget'] = pd.to_numeric(ml_df['budget'], errors='coerce').fillna(0)
ml_df['revenue'] = pd.to_numeric(ml_df['revenue'], errors='coerce').fillna(0)
ml_df['popularity'] = pd.to_numeric(ml_df['popularity'], errors='coerce').fillna(0)

# Filter out movies with 0 budget or 0 revenue for clean training
clean_df = ml_df[(ml_df['budget'] > 0) & (ml_df['revenue'] > 0)].copy()

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("Performing Exploratory Data Analysis...")

# Correlation Matrix
plt.figure(figsize=(10, 8))
corr_matrix = clean_df[['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'revenue']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Movie Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png') # Saves the plot so you can view it
print("Correlation matrix saved as 'correlation_matrix.png'")

# ==========================================
# 3. FEATURE ENGINEERING & TARGET CREATION
# ==========================================
# Instead of exact revenue, we categorize revenue into 3 Tiers for Logistic Regression
# Tier 0: Low Revenue, Tier 1: Medium Revenue, Tier 2: High Revenue
print("Categorizing Revenue into Tiers...")
quantiles = clean_df['revenue'].quantile([0.33, 0.66])
def categorize_revenue(rev):
    if rev <= quantiles.iloc[0]:
        return 0 # Low
    elif rev <= quantiles.iloc[1]:
        return 1 # Medium
    else:
        return 2 # High

clean_df['revenue_tier'] = clean_df['revenue'].apply(categorize_revenue)

# Define Features (More than just the 3 you had before!)
features = ['budget', 'runtime', 'popularity', 'vote_average', 'vote_count']
X = clean_df[features]
y = clean_df['revenue_tier']

# ==========================================
# 4. MODEL TRAINING (LOGISTIC REGRESSION)
# ==========================================
print("Training Logistic Regression Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Logistic Regression
# using 'saga' solver as it handles multinomial (multiple categories) well
log_reg = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# ==========================================
# 5. MODEL EVALUATION
# ==========================================
print("\nEvaluating Model...")
y_pred = log_reg.predict(X_test_scaled)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Low Rev', 'Med Rev', 'High Rev']))

print("\n--- Confusion Matrix ---")
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

# Visualizing the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted Revenue Tier')
plt.ylabel('Actual Revenue Tier')
plt.title('Confusion Matrix - Revenue Tier Predictor')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix plot saved as 'confusion_matrix.png'")

# ==========================================
# 6. EXPORT MODELS (OVERWRITING OLD ONES)
# ==========================================
# Overwriting the existing pickle names so we don't create new ones
pickle.dump(log_reg, open('models/rf_classifier.pkl', 'wb'))
pickle.dump(scaler, open('models/ml_scaler.pkl', 'wb'))

# Save the cleaned dataset for the dashboard
clean_df.to_pickle('models/clustered_movies.pkl')

print("\nSuccess! Models have overwritten the existing pickle files.")