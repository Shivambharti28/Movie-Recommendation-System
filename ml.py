import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the original dataset
df = pd.read_csv('AI_movies_dataset.csv')

# 2. Data Quality Assessment (Syllabus Unit II)
# Keep relevant columns for ML
ml_df = df[['title', 'budget', 'revenue', 'runtime', 'vote_average', 'popularity']].copy()

# Drop rows where title is missing and fill missing numbers with 0 or mean
ml_df = ml_df.dropna(subset=['title'])
ml_df['runtime'] = ml_df['runtime'].fillna(ml_df['runtime'].mean())
ml_df['budget'] = pd.to_numeric(ml_df['budget'], errors='coerce').fillna(0)
ml_df['revenue'] = pd.to_numeric(ml_df['revenue'], errors='coerce').fillna(0)

# Filter out movies with 0 budget or 0 revenue to get a clean training set
clean_df = ml_df[(ml_df['budget'] > 0) & (ml_df['revenue'] > 0)].copy()

# ==========================================
# IDEA 1: THE BLOCKBUSTER PREDICTOR (CLASSIFICATION)
# Syllabus Unit IV
# ==========================================

# Define our target label: Is it a commercial hit?
# Let's define a "Hit" as a movie that made at least 2.5x its budget
clean_df['is_hit'] = (clean_df['revenue'] >= (clean_df['budget'] * 2.5)).astype(int)

# Features and Target
X_class = clean_df[['budget', 'runtime', 'popularity']]
y_class = clean_df['is_hit']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Evaluate (Syllabus Unit VI)
y_pred = rf_classifier.predict(X_test_scaled)
print("Classification Report for Blockbuster Predictor:")
print(classification_report(y_test, y_pred))

# ==========================================
# IDEA 2: MOVIE UNIVERSE VISUALIZER (CLUSTERING)
# Syllabus Unit V
# ==========================================

# Use PCA to reduce dimensions for 2D visualization
features_to_cluster = clean_df[['budget', 'revenue', 'runtime', 'vote_average']]
features_scaled = StandardScaler().fit_transform(features_to_cluster)

pca = PCA(n_components=2)
components = pca.fit_transform(features_scaled)
clean_df['pca_x'] = components[:, 0]
clean_df['pca_y'] = components[:, 1]

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clean_df['cluster'] = kmeans.fit_predict(features_scaled)

# ==========================================
# EXPORT MODELS
# ==========================================
pickle.dump(rf_classifier, open('rf_classifier.pkl', 'wb'))
pickle.dump(scaler, open('ml_scaler.pkl', 'wb'))
clean_df.to_pickle('clustered_movies.pkl')

print("Models successfully saved!")