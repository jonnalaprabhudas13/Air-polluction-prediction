import os
from django.conf import settings
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np

class AirQualityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def get_data_path(self):
        """Get the absolute path to the data file"""
        data_dir = os.path.join(settings.BASE_DIR, 'predictor', 'data')
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, 'updated_pollution_dataset.csv')
    
    def get_model_path(self):
        """Get the absolute path to the model file"""
        model_dir = os.path.join(settings.BASE_DIR, 'predictor', 'models')
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, 'air_quality_model.pkl')
    
    def get_scaler_path(self):
        """Get the absolute path to the scaler file"""
        model_dir = os.path.join(settings.BASE_DIR, 'predictor', 'models')
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, 'scaler.pkl')
    
    def load_model(self):
        """Load or create the model"""
        data_path = self.get_data_path()
        model_path = self.get_model_path()
        scaler_path = self.get_scaler_path()
        
        # Check if data file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset file not found at {data_path}. "
                "Please ensure the CSV file is in the predictor/data directory."
            )
        
        # Load or train model
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            self.train_and_save_model()
    
    def train_and_save_model(self):
        """Train and save the model"""
        # Load data
        df = pd.read_csv(self.get_data_path())
        
        # Preprocess data
        df['Air Quality'] = pd.factorize(df['Air Quality'])[0]
        X = df.drop(['Air Quality'], axis=1).values
        y = df['Air Quality'].values
        
        # Split and scale data
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        self.scaler = MinMaxScaler().fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Train model
        param = [1, 2]
        clf = SVC(random_state=0, probability=True)
        self.model = GridSearchCV(estimator=clf, param_grid={'C': param}, cv=10).fit(X_train_scaled, y_train)
        self.model = self.model.best_estimator_
        self.model.fit(X_train_scaled, y_train)
        
        # Save model
        joblib.dump(self.model, self.get_model_path())
        joblib.dump(self.scaler, self.get_scaler_path())
    
    def predict(self, input_data):
        """Make a prediction"""
        try:
            input_array = np.array([[
                input_data['temperature'],
                input_data['humidity'],
                input_data['pm25'],
                input_data['pm10'],
                input_data['no2'],
                input_data['so2'],
                input_data['co'],
                input_data['proximity'],
                input_data['population_density']
            ]])
            
            scaled_input = self.scaler.transform(input_array)
            prediction = self.model.predict(scaled_input)[0]
            proba = self.model.predict_proba(scaled_input)[0]
            
            quality_labels = ['Moderate', 'Good', 'Hazardous']
            quality = quality_labels[prediction]
            
            return {
                'quality': quality,
                'probability': round(max(proba) * 100, 2),
                'details': dict(zip(quality_labels, [round(p*100, 2) for p in proba]))
            }
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None










# from django.db import models

# # Create your models here.

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# import joblib
# import os
# from django.conf import settings

# class AirQualityPredictor:
#     def __init__(self):
#         self.model = None
#         self.scaler = None
#         self.load_model()
        
#     def load_model(self):
#         model_path = os.path.join(settings.BASE_DIR, 'predictor', 'models', 'air_quality_model.pkl')
#         scaler_path = os.path.join(settings.BASE_DIR, 'predictor', 'models', 'scaler.pkl')
        
#         if os.path.exists(model_path) and os.path.exists(scaler_path):
#             self.model = joblib.load(model_path)
#             self.scaler = joblib.load(scaler_path)
#         else:
#             self.train_and_save_model()
    
#     def train_and_save_model(self):
#         # Load and preprocess data
#         df = pd.read_csv(os.path.join(settings.BASE_DIR, 'predictor', 'data', 'updated_pollution_dataset.csv'))
        
#         # Convert categorical 'Air Quality' to numerical
#         df['Air Quality'] = pd.factorize(df['Air Quality'])[0]
        
#         X = df.drop(['Air Quality'], axis=1).values
#         y = df['Air Quality'].values
        
#         # Split and scale data
#         X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
#         self.scaler = MinMaxScaler().fit(X_train)
#         X_train_scaled = self.scaler.transform(X_train)
        
#         # Train model
#         param = [1, 2]
#         clf = SVC(random_state=0, probability=True)
#         self.model = GridSearchCV(estimator=clf, param_grid={'C': param}, cv=10).fit(X_train_scaled, y_train)
#         self.model = self.model.best_estimator_
#         self.model.fit(X_train_scaled, y_train)
        
#         # Save model and scaler
#         os.makedirs(os.path.join(settings.BASE_DIR, 'predictor', 'models'), exist_ok=True)
#         joblib.dump(self.model, os.path.join(settings.BASE_DIR, 'predictor', 'models', 'air_quality_model.pkl'))
#         joblib.dump(self.scaler, os.path.join(settings.BASE_DIR, 'predictor', 'models', 'scaler.pkl'))
    
#     def predict(self, input_data):
#         try:
#             # Convert input data to numpy array and scale
#             input_array = np.array([[
#                 input_data['temperature'],
#                 input_data['humidity'],
#                 input_data['pm25'],
#                 input_data['pm10'],
#                 input_data['no2'],
#                 input_data['so2'],
#                 input_data['co'],
#                 input_data['proximity'],
#                 input_data['population_density']
#             ]])
            
#             scaled_input = self.scaler.transform(input_array)
            
#             # Make prediction
#             prediction = self.model.predict(scaled_input)[0]
#             proba = self.model.predict_proba(scaled_input)[0]
            
#             # Map prediction to quality label
#             quality_labels = ['Moderate', 'Good', 'Hazardous']
#             quality = quality_labels[prediction]
            
#             return {
#                 'quality': quality,
#                 'probability': round(max(proba) * 100, 2),
#                 'details': dict(zip(quality_labels, [round(p*100, 2) for p in proba]))
#             }
#         except Exception as e:
#             print(f"Prediction error: {str(e)}")
#             return None