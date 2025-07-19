# import libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load dataset
df = pd.read_csv("data/Salary Data.csv")
df.dropna(inplace=True)

# Feature Engineering
df["Experience_per_Age"] = df["Years of Experience"] / df["Age"]
df["Seniority"] = df["Job Title"].apply(lambda x: 1 if any(w in x for w in ["Senior", "Manager", "Director", "Lead"]) else 0)

# Target and Features
y = np.log1p(df["Salary"])
X = df.drop("Salary", axis=1)

# Feature Types
ordinal_features = ["Education Level"]
nominal_features = ["Gender", "Job Title"]
numeric_features = ["Age", "Years of Experience", "Experience_per_Age", "Seniority"]

# Pipelines
ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

nominal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])

preprocessor = ColumnTransformer([
    ("ord", ordinal_pipeline, ordinal_features),
    ("nom", nominal_pipeline, nominal_features),
    ("num", numeric_pipeline, numeric_features)
])

# Final model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', random_state=28))
])

# Grid Search Parameters
param_grid = {
    'regressor__n_estimators': [500, 700],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__max_depth': [4, 6, 8],
    'regressor__subsample': [0.8, 1.0],
    'regressor__colsample_bytree': [0.7, 1.0]
}

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# Model Training
grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=2)
grid_search.fit(pd.DataFrame(X_train, columns=X.columns), y_train)


# Evaluation
y_pred_log = grid_search.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)
residuals = y_actual - y_pred

r2 = r2_score(y_actual, y_pred)*100
rmse = mean_squared_error(y_actual, y_pred) ** 0.5
print("\n✅ R² Score:", round(r2, 4))
print("✅ RMSE: $", round(rmse, 2))
print("✅ Best Parameters:", grid_search.best_params_)

# Safely get feature names from ColumnTransformer
def get_feature_names(preprocessor):
    try:
        output_features = []

        for name, transformer, columns in preprocessor.transformers_:
            if transformer == 'drop' or transformer is None:
                continue

            # Case 1: pipeline
            if hasattr(transformer, 'named_steps'):
                try:
                    last_step = list(transformer.named_steps.values())[-1]
                    if hasattr(last_step, 'get_feature_names_out'):
                        names = last_step.get_feature_names_out(columns)
                        output_features.extend(names)
                    else:
                        output_features.extend(columns)
                except Exception as e:
                    print(f"⚠️ Skipping pipeline transformer {name} due to error: {e}")
                    output_features.extend(columns)

            # Case 2: direct transformer
            elif hasattr(transformer, 'get_feature_names_out'):
                try:
                    names = transformer.get_feature_names_out(columns)
                    output_features.extend(names)
                except Exception as e:
                    print(f"⚠️ Skipping transformer {name} due to error: {e}")
                    output_features.extend(columns)

            else:
                output_features.extend(columns)

        return output_features

    except Exception as e:
        print("🚫 ERROR in get_feature_names:", e)
        return []  # <- return empty list instead of None


# Get feature names
feature_names = get_feature_names(grid_search.best_estimator_.named_steps["preprocessor"])

# Map XGBoost feature importances
booster = grid_search.best_estimator_.named_steps["regressor"].get_booster()
xgb_features = booster.feature_names
print("DEBUG - feature_names is:", feature_names)
print("DEBUG - xgb_features is:", xgb_features)
print("DEBUG - length match:", len(feature_names) if feature_names else None,
      len(xgb_features) if xgb_features is not None else None)

if not feature_names or not xgb_features or len(feature_names) != len(xgb_features):
    print("⚠️ Feature names missing or mismatch — using default fallback names.")
    feature_mapping = {f: f for f in (xgb_features or [])}
else:
    feature_mapping = {xgb_feat: real_name for xgb_feat, real_name in zip(xgb_features, feature_names)}


importance_dict = booster.get_score(importance_type='gain')
sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\n🔎 Top 15 Most Important Features:")
for feat, score in sorted_features[:15]:
    print(f"{feature_mapping.get(feat, feat)}: {round(score, 4)}")

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("📊 Model Evaluation Plots", fontsize=20, fontweight='bold')

# 1. Actual vs Predicted Line Plot
axs[0, 0].plot(np.arange(50), y_actual[:50], label='Actual', marker='o')
axs[0, 0].plot(np.arange(50), y_pred[:50], label='Predicted', marker='x')
axs[0, 0].set_title("Actual vs Predicted Salary (Sample 50)")
axs[0, 0].set_xlabel("Sample Index")
axs[0, 0].set_ylabel("Salary")
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. Residual Histogram
sns.histplot(residuals, kde=True, bins=30, ax=axs[0, 1], color='skyblue')
axs[0, 1].set_title("Distribution of Residuals")
axs[0, 1].set_xlabel("Residuals")

# 3. Feature Importance
top15 = sorted_features[:15]
top15_labels = [feature_mapping.get(f, f) for f, _ in top15]
top15_scores = [s for _, s in top15]
axs[1, 1].barh(top15_labels[::-1], top15_scores[::-1], color='purple')
axs[1, 1].set_title("Feature Importance (Top 15)")
axs[1, 1].set_xlabel("Importance")
axs[1, 1].grid(True)

# 4. Scatter: Actual vs Predicted
axs[1, 0].scatter(y_actual, y_pred, alpha=0.7)
axs[1, 0].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
axs[1, 0].set_title("Scatter Plot: Actual vs Predicted")
axs[1, 0].set_xlabel("Actual Salary")
axs[1, 0].set_ylabel("Predicted Salary")
axs[1, 0].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(grid_search.best_estimator_, "models/salary_model.pkl")
print("✅ Model saved to models/salary_model.pkl")
