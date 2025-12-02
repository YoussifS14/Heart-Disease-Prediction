from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QComboBox, QGridLayout, QScrollArea, QGroupBox
)
from PyQt5.QtGui import QIcon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('heart.csv')
for col in data.columns:
    if data[col].dtype == object:
        data[col] = pd.to_numeric(data[col], errors='coerce')

data.drop_duplicates(inplace=True)
outlier = ['trestbps', 'chol', 'thalach', 'oldpeak']
numerical_cols = ['age']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for col in outlier:
    data[col] = data[col].fillna(data[col].median())
for col in numerical_cols:
    mean_value = int(data[col].mean())
    data[col] = data[col].fillna(mean_value)
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])
for col in outlier:
    q1 = np.percentile(data[col], 25)
    q3 = np.percentile(data[col], 75)
    iqr = (q3 - q1) * 1.5
    data[col] = np.where(data[col] < (q1 - iqr), q1 - iqr, data[col])
    data[col] = np.where(data[col] > (q3 + iqr), q3 + iqr, data[col])
data['ca'] = data['ca'].clip(lower=0, upper=3)
data.to_csv("cleaned_heart_data.csv", index=False)

features_used = ['cp', 'thalach', 'exang', 'oldpeak','ca']
X = data[features_used]
Y = data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression( C=0.016,
    max_iter=100,),
    "SVM": SVC(kernel='linear', probability=True,C=0.1),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_leaf=8,min_samples_split=2,max_leaf_nodes=9),
    "KNN": KNeighborsClassifier(n_neighbors=14)}
for model in models.values():
    model.fit(X_train, Y_train)

class HeartDiseaseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Disease Prediction System")
        self.setWindowIcon(QIcon("download.png"))  # Ensure the icon file exists
        self.inputs = {}
        self.init_ui()
    def init_ui(self):
        main_layout = QVBoxLayout()
        title = QLabel("🩺 Heart Disease Prediction")
        title.setStyleSheet("font-size: 22px; font-weight: bold; margin: 10px; text-align: center;")
        main_layout.addWidget(title)
        self.fields = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak',
            'slope', 'ca', 'thal'
        ]
        combo_fields = {
            'sex': ['0 - Female', '1 - Male'],
            'cp': ['0 - Typical Angina', '1 - Atypical Angina', '2 - Non-anginal Pain', '3 - Asymptomatic'],
            'fbs': ['0 - False', '1 - True'],
            'restecg': ['0 - Normal', '1 - ST-T wave abnormality', '2 - Left ventricular hypertrophy'],
            'exang': ['0 - No', '1 - Yes'],
            'slope': ['0 - Upsloping', '1 - Flat', '2 - Downsloping'],
            'ca': ['0', '1', '2', '3'],
            'thal': ['1 - Normal', '2 - Fixed defect', '3 - Reversible defect']
        }
        form_layout = QGridLayout()
        form_layout.setHorizontalSpacing(10)
        form_layout.setVerticalSpacing(4)
        for i, field in enumerate(self.fields):
            label = QLabel(f"{field} ({self.get_description(field)}):")
            label.setStyleSheet("padding: 2px;")
            if field in combo_fields:
                combo = QComboBox()
                combo.addItems(combo_fields[field])
                self.inputs[field] = combo
                form_layout.addWidget(label, i, 0)
                form_layout.addWidget(combo, i, 1)
            else:
                edit = QLineEdit()
                edit.setPlaceholderText(f"Enter {field}")
                self.inputs[field] = edit
                form_layout.addWidget(label, i, 0)
                form_layout.addWidget(edit, i, 1)
        form_group = QGroupBox("🔢 Enter All Clinical Features")
        form_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; margin-top: 10px; }")
        form_group.setLayout(form_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_group)
        main_layout.addWidget(scroll)
        model_label = QLabel("🔽 Choose Model:")
        model_label.setStyleSheet("margin-top: 8px; font-weight: bold;")
        self.model_box = QComboBox()
        self.model_box.addItems(models.keys())
        main_layout.addWidget(model_label)
        main_layout.addWidget(self.model_box)
        button_layout = QHBoxLayout()
        predict_btn = QPushButton("🔍 Predict")
        clear_btn = QPushButton("🧹 Clear")
        predict_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 6px;")
        clear_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 6px;")
        predict_btn.clicked.connect(self.predict)
        clear_btn.clicked.connect(self.clear_inputs)
        button_layout.addWidget(predict_btn)
        button_layout.addWidget(clear_btn)
        main_layout.addLayout(button_layout)
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("font-size: 16px; margin-top: 15px;")
        main_layout.addWidget(self.result_label)
        self.setLayout(main_layout)
        self.setMinimumWidth(550)
    def get_description(self, field):
        descriptions = {
            'age': 'Age in years',
            'sex': '0=Female, 1=Male',
            'cp': 'Chest Pain Type (0–3)',
            'trestbps': 'Resting Blood Pressure',
            'chol': 'Serum Cholesterol',
            'fbs': 'Fasting Blood Sugar > 120 (0/1)',
            'restecg': 'Resting ECG (0–2)',
            'thalach': 'Max Heart Rate',
            'exang': 'Exercise Angina (0/1)',
            'oldpeak': 'ST Depression',
            'slope': 'Slope of ST (0–2)',
            'ca': 'No. of vessels (0–3)',
            'thal': 'Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)'
        }
        return descriptions.get(field, '')
    def predict(self):
        try:
            user_data = {}
            for f in self.fields:
                widget = self.inputs[f]
                if isinstance(widget, QComboBox):
                    text = widget.currentText().split(' - ')[0]
                else:
                    text = widget.text()
                if not text.strip().replace('.', '', 1).replace('-', '', 1).isdigit():
                    raise ValueError(f"Invalid input for '{f}': must be a number.")
                user_data[f] = float(text)
            selected_features = [user_data[f] for f in features_used]
            selected_scaled = scaler.transform([selected_features])
            model_name = self.model_box.currentText()
            model = models[model_name]
            prediction = model.predict(selected_scaled)[0]
            proba = model.predict_proba(selected_scaled)[0][1] if hasattr(model, "predict_proba") else None
            if prediction == 1:
                self.result_label.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
                self.result_label.setText(f"⚠️ High Risk. Yes Heart Disease.\nProbability: {proba:.2%}")
            else:
                self.result_label.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
                self.result_label.setText(f"✅ Low Risk. No Heart Disease.\nProbability: {proba:.2%}")
        except Exception as e:
            self.result_label.setStyleSheet("color: orange; font-size: 14px;")
            self.result_label.setText(f"❌ Error: {str(e)}")
    def clear_inputs(self):
        for field in self.fields:
            widget = self.inputs[field]
            if isinstance(widget, QLineEdit):
                widget.clear()
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(0)
        self.result_label.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeartDiseaseApp()
    window.show()
    sys.exit(app.exec_())