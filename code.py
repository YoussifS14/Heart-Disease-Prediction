# **Libraries**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from sklearn.model_selection import  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  roc_curve, auc
from sklearn import metrics
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.metrics import (classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, mean_squared_error,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
import warnings
from sklearn.model_selection import  StratifiedKFold
warnings.filterwarnings('ignore')

# **Read data & information**

data = pd.read_csv('heart.csv')

data.info()

data.describe(include="all")

data.columns

data.isnull().sum()

data.isnull().sum().sort_values(ascending = 0)

data.duplicated().sum()

data.head(10)

data.tail(10)

# **Convert to numerical**

for col in data.columns:
  if data[col].dtype == object:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data.describe(include="all")

# **Data visualization**

plt.figure(figsize=(12,6))
sns.boxplot(data=data)
plt.xticks(rotation=45)
plt.title("Box Plot for Outlier Detection")
plt.show()

print("Columns in data:", data.columns.tolist())
for col in data.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[col], kde=False, stat='density', bins=20, color='skyblue', edgecolor='black')
    mean = data[col].mean()
    std = data[col].std()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, 'r', linewidth=2)
    plt.title(f'Histogram and Normal Curve of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True)
    plt.legend(['Normal Distribution', 'Data Histogram'])
    plt.show()

sns.pairplot(data)
plt.show()

for col in data.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=data[col], color='lightgreen')
    plt.title(f'Boxplot for {col}')
    plt.grid(True)
    plt.show()

missingno.matrix(data)

data.isnull().sum()

# **Handling null , missing data & duplicated**

data.duplicated().sum()

data.drop_duplicates(inplace=True)

data.duplicated().sum()

missingno.matrix(data)

plt.figure(figsize=(12,6))
sns.boxplot(data=data)
plt.xticks(rotation=45)
plt.title("Box Plot for Detection")
plt.show()

data.columns

outlier = ['trestbps','chol','thalach','oldpeak']
numerical_cols = ['age']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca','thal']

for col in outlier:
  data[col] = data[col].fillna(data[col].median())

for col in numerical_cols:
    mean_value = int(data[col].mean())
    data[col] = data[col].fillna(mean_value)

for col in categorical_cols:
 data[col] = data[col].fillna(data[col].mode()[0])

data['ca'] = data['ca'].clip(lower=0, upper=3)

missingno.matrix(data)

data.isnull().sum()

data.info()

# **outliers**

plt.figure(figsize=(12,6))
sns.boxplot(data=data)
plt.xticks(rotation=45)
plt.title("Box Plot for  Detection")
plt.show()

for col in outlier:
 q1 = np.percentile(data[col], 25)
 q3 = np.percentile(data[col], 75)
 print(f"Q1 for {col} : {q1}")
 print(f"Q3 for {col} : {q3}")
 norm_range = (q3 - q1) * 1.5
 lower_outliers = data[data[col] < (q1 - norm_range)]
 upper_outliers = data[data[col] > (q3 + norm_range)]
 print(f"lower_outliers for {col} : {q1-norm_range}")
 print(f"uper_outliers for {col} : {q3+norm_range}")
 outliers = len(lower_outliers)+len(upper_outliers)
 print(f"The number of outliers in {col} :  {outliers}")
 print()

for col in outlier:
   q1 = np.percentile(data[col], 25)
   q3 = np.percentile(data[col], 75)
   norm_range = (q3 - q1) * 1.5
   data[col] = np.where(data[col] < (q1 - norm_range), q1 - norm_range, data[col])
   data[col] = np.where(data[col] > (q3 + norm_range), q3 + norm_range, data[col])
   lower_outliers = data[data[col] < (q1 - norm_range)]
   upper_outliers = data[data[col] > (q3 + norm_range)]
   outliers = len(lower_outliers)+len(upper_outliers)
   print(f"The number of outliers in {col} :  {outliers}")

plt.figure(figsize=(12, 6))
sns.boxplot(data=data[outlier])
plt.xticks(rotation=45)
plt.title("Box Plot for Outlier Detection")
plt.show()

data.to_csv("cleaned_heart_data.csv", index=False)

# **Correlation**

high_corr = []
low_corr = []
bad_corr = []
for col in data.columns:
  if col == 'target':
    continue
  relation = data['target'].corr(data[col])
  if(relation > 0):
    if relation >= 0.7 and relation < 0.95 :
      high_corr.append(col)
    elif relation >= 0.4 and relation < 0.7 :
      low_corr.append(col)
    else: bad_corr.append(col)
  else:
    if relation <= -0.7 and relation > -0.95 :
      high_corr.append(col)
    elif relation <= -0.4 and relation > -0.7 :
      low_corr.append(col)
    else: bad_corr.append(col)
print(f"the high corr are {high_corr}")
print(f"the low corr are {low_corr}")
print(f"the bad corr are {bad_corr}")

data.corr()

data.corr()['target']

# **Visualize correlation**

fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(),annot = True, ax =ax)

corr_with_target = data.corr()['target'].drop('target').to_frame()
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corr_with_target, annot=True, cmap='coolwarm', cbar=True, ax=ax)
ax.set_title("Correlation of Features with Target ", fontsize=16)
plt.show()

sns.pairplot(data, hue='target', height=2.2, corner=True)

correlations = data.corr()['target'].drop('target')
correlations.sort_values().plot(kind='barh', figsize=(8,6), color='skyblue')
plt.title('Correlation with Target')
plt.xlabel('Correlation')
plt.grid(True)

# **Models**

##  **data**

X = data[high_corr + low_corr]
Y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train class distribution:", np.bincount(y_train))
print("Test class distribution:", np.bincount(y_test))

unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))
print(y_train.value_counts())

##  **Logistic Regression**

log_reg = LogisticRegression()
param_grid = {
    'C': [0.001, 0.016, 0.1, 1, 10, 100, 1000],
    'max_iter': [100, 200, 300, 400, 500]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:")
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

logistic_model = LogisticRegression(
    C=0.016229876459045892,
    max_iter=100,
)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]
auc_log = roc_auc_score(y_test, y_pred_proba)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logistic_model.score(X_test, y_test)))
print(f'Logistic Regression AUC: {auc_log:.2f}')
print('Co-efficient of logistic regression:', logistic_model.coef_)
print('Intercept of logistic regression model:', logistic_model.intercept_)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
true_value = y_test.iloc[0]
predicted_value = y_pred[0]
print('True value for the first test sample: ' + str(true_value))
print('Predicted value for the first test sample: ' + str(predicted_value))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
confusion_matrix1 = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix1, annot=True, cmap="Blues", fmt='d')
plt.show()

## **SVM**

svm = SVC(kernel='linear', probability=True, random_state=42)
param_grid = {
    'C': np.logspace(-1, 1, 10)
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:")
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

svm_model = SVC(kernel='linear', probability=True,C=0.1)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
y_pred_proba = svm_model.predict_proba(X_test)[:, 1]
auc_log = roc_auc_score(y_test, y_pred_proba)
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm_model.score(X_test, y_test)))
print(f'SVM AUC: {auc_log:.2f}')
print('Co-efficient of SVM (only for linear kernel):', svm_model.coef_)
print('Intercept of SVM model:', svm_model.intercept_)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
true_value = y_test.iloc[0] if hasattr(y_test, 'iloc') else y_test[0]
predicted_value = y_pred[0]
print('True value for the first test sample:', true_value)
print('Predicted value for the first test sample:', predicted_value)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
confusion_matrix2 = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix2, annot=True, cmap="Blues", fmt='d')
plt.show()

## **Decision Tree**

tree = DecisionTreeClassifier(criterion='entropy')
param_grid = {
    'max_depth': [None, 3, 5, 7, 10, 15,1,2,20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4,5,8,9,10],
     'max_leaf_nodes': [None, 5, 7, 9, 10]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:")
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

tree_model = DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_leaf=8,min_samples_split=2,max_leaf_nodes=9)
tree_model.fit(X_train, y_train)
y_pred =tree_model.predict(X_test)
y_pred_proba = tree_model.predict_proba(X_test)[:, 1]
auc_dt = roc_auc_score(y_test, y_pred_proba)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(tree_model.score(X_test, y_test)))
print(f'Decision Tree AUC: {auc_dt:.2f}')
print('Feature Importances of the Decision Tree:', tree_model.feature_importances_)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
true_value = y_test.iloc[0] if hasattr(y_test, 'iloc') else y_test[0]
predicted_value = y_pred[0]
print('True value for the first test sample:', true_value)
print('Predicted value for the first test sample:', predicted_value)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
confusion_matrix3 = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix3, annot=True, cmap="Blues", fmt='d')
plt.show()

## **KNN**

knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [1,3,5,14,7,9,20,2]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:")
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

knn_model = KNeighborsClassifier(n_neighbors=14)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
y_pred_proba = knn_model.predict_proba(X_test)[:, 1]
auc_knn = roc_auc_score(y_test, y_pred_proba)
print('Accuracy of KNN classifier on test set: {:.2f}'.format(knn_model.score(X_test, y_test)))
print(f'KNN AUC: {auc_knn:.2f}')
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
true_value = y_test.iloc[0] if hasattr(y_test, 'iloc') else y_test[0]
predicted_value = y_pred[0]
print('True value for the first test sample:', true_value)
print('Predicted value for the first test sample:', predicted_value)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
confusion_matrix4 = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix4, annot=True, cmap="Blues", fmt='d')
plt.show()

# **Visualize**

models = {
    'Logistic Regression':LogisticRegression(
    C=0.016229876459045892,
    max_iter=100,
),
    'SVM': SVC(kernel='linear', probability=True,C=0.1),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_leaf=8,min_samples_split=2,max_leaf_nodes=9),
    'KNN': KNeighborsClassifier(n_neighbors=14)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Training Accuracy: {train_accuracy * 100:.2f} %")
    print(f"Test Accuracy: {test_accuracy * 100:.2f} %")
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    results[name] = {'accuracy': acc, 'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr}
    print(f"{name} ROC AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    confusion_matrix4 = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix4, annot=True, cmap="Blues", fmt='d')
    plt.show()

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    if name == 'SVM':
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_model = roc_auc_score(y_test, y_prob)
    else:
        auc_model = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    print(f"\n{name} ")
    print(f"Accuracy: {acc:.4f}")
    print(f'{name} Roc AUC: {auc_model:.2f}')
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
    accuracies = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

plt.figure(figsize=(10, 6))
bars = plt.bar(models.keys(), accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison - Accuracy')
plt.ylim(0, 1)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')
plt.show()

plt.figure(figsize=(8, 6))
for name, res in results.items():
    plt.scatter(
        res['precision'],
        res['recall'],
        label=f"{name} (Accuracy = {res['accuracy']:.2f})",
        s=100
    )
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-Recall Comparison')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.show()

x_axis = np.arange(len(models))
for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    plt.plot(x_axis[idx], acc, 'o-', markersize=10, label=f"{name} (Acc = {acc:.2f})")
plt.xticks(x_axis, models.keys())
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison - Accuracy')
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]
y_score_svm = svm_model.decision_function(X_test)
y_score_dt = tree_model.predict_proba(X_test)[:, 1]
y_score_knn = knn_model.predict_proba(X_test)[:, 1]
fpr_log, tpr_log, _ = roc_curve(y_test, y_score_logistic)
auc_log = roc_auc_score(y_test, y_score_logistic)
print(f'Logistic Regression AUC: {auc_log:.2f}')
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
auc_svm = roc_auc_score(y_test, y_score_svm)
print(f'SVM AUC: {auc_svm:.2f}')
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_score_dt)
auc_dt = roc_auc_score(y_test, y_score_dt)
print(f'Decision Tree AUC: {auc_dt:.2f}')
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_score_knn)
auc_knn = roc_auc_score(y_test, y_score_knn)
print(f'KNN AUC: {auc_knn:.2f}')
plt.figure(figsize=(10, 8))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_log:.2f})', color='blue')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.2f})', color='green')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})', color='red')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {auc_knn:.2f})', color='purple')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

comparison_df = pd.DataFrame({
    'model': ['Logistic Regression', 'SVM', 'Decision Tree', 'KNN'],
    'accuracy': [
        accuracy_score(y_test, logistic_model.predict(X_test)),
        accuracy_score(y_test, svm_model.predict(X_test)),
        accuracy_score(y_test, tree_model.predict(X_test)),
        accuracy_score(y_test, knn_model.predict(X_test))
    ],
    'AUC': [
        auc(*roc_curve(y_test, logistic_model.predict_proba(X_test)[:, 1])[:2]),
        auc(*roc_curve(y_test, svm_model.decision_function(X_test))[:2]),
        auc(*roc_curve(y_test, tree_model.predict_proba(X_test)[:, 1])[:2]),
        auc(*roc_curve(y_test, knn_model.predict_proba(X_test)[:, 1])[:2])
        ]
})
print(comparison_df.to_string(index=False))

# **bonus**

## **feature selection**

def evaluate_model(model, X, Y, **params):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_accuracy = 0
    best_k = 0
    for k in range(1, 14):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_kbest = selector.fit_transform(X_scaled, Y)
        X_train_k, X_test_k, Y_train_k, Y_test_k = train_test_split(X_kbest, Y, test_size=0.2, random_state=42)
        model_instance = model(**params)
        model_instance.fit(X_train_k, Y_train_k)
        Y_pred_k = model_instance.predict(X_test_k)
        acc = accuracy_score(Y_test_k, Y_pred_k)
        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k
    print(f"Best accuracy for {model.__name__}: {best_accuracy:.4f} with top {best_k} features\n")
models_params = [
    (LogisticRegression, {'max_iter': 100,'C':0.016229876459045892}),
    (SVC, {'kernel': 'linear', 'probability': True,'C':.1}),
    (DecisionTreeClassifier, {'criterion': 'entropy','max_depth':None,'min_samples_leaf':8,'min_samples_split':2,'max_leaf_nodes':9}),
    (KNeighborsClassifier, {'n_neighbors': 14}),
]
for model, params in models_params:
    evaluate_model(model, X, Y, **params)

models_params = [
    (LogisticRegression, {'max_iter': 100,'C':0.016229876459045892}, 5),
    (SVC, {'kernel': 'linear', 'probability': True,'C':.1}, 5),
    (DecisionTreeClassifier, {'criterion': 'entropy','max_depth':None,'min_samples_leaf':8,'min_samples_split':2,'max_leaf_nodes':9}, 4),
    (KNeighborsClassifier, {'n_neighbors': 14}, 5),
]

def evaluate_model_with_feature_extraction(model, X, Y, k, use_pca=False):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"\n===== Model: {model.__class__.__name__}, Selected Features: {k} =====")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if use_pca:
        pca = PCA(n_components=k)
        X_train_sel = pca.fit_transform(X_train_scaled)
        X_test_sel = pca.transform(X_test_scaled)
        print(f"Using PCA with {k} components.")
    else:
        selector = RFE(model, n_features_to_select=k)
        selector.fit(X_train_scaled, Y_train)
        selected_features = X.columns[selector.support_]
        X_train_sel = pd.DataFrame(X_train_scaled, columns=X.columns)[selected_features]
        X_test_sel = pd.DataFrame(X_test_scaled, columns=X.columns)[selected_features]
        print(f"Using RFE with {k} features. Selected features: {selected_features.tolist()}")
    model.fit(X_train_sel, Y_train)
    y_pred = model.predict(X_test_sel)
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    cm = confusion_matrix(Y_test, y_pred)
    auc = None
    if hasattr(model, 'predict_proba'):
        try:
            auc = roc_auc_score(Y_test, model.predict_proba(X_test_sel)[:, 1])
        except:
            auc = None
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    print("\nClassification Report:\n", classification_report(Y_test, y_pred))
    return {
        "Model": model.__class__.__name__,
        "K Features": k,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MSE": mse,
        "Confusion Matrix": cm.tolist(),
        "AUC": auc
    }
models_params = [
    (LogisticRegression, {'max_iter': 100,'C':0.016229876459045892}, 5),
    (SVC, {'kernel': 'linear', 'probability': True,'C':.1}, 5),
    (DecisionTreeClassifier, {'criterion': 'entropy','max_depth':None,'min_samples_leaf':8,'min_samples_split':2,'max_leaf_nodes':9}, 4),
    (KNeighborsClassifier, {'n_neighbors': 14}, 5),
]
for model_class, params, k in models_params:
    model = model_class(**params)
    evaluate_model_with_feature_extraction(model, X, Y, k)

## **classification algorithms**

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def fit(self, X, y):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    def predict(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i >= 0.5 else 0 for i in y_predicted]

model = LogisticRegressionScratch(lr=0.016, n_iters=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))