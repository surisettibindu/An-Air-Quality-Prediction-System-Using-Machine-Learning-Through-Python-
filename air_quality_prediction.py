#!/usr/bin/env python3
"""
Air Quality Prediction System Using Machine Learning
This script implements a comprehensive air quality prediction system using various ML algorithms.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class AirQualityPredictor:
    """
    A comprehensive air quality prediction system using machine learning.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        self.target_name = ""
        
    def load_data(self, file_path=None):
        """
        Load air quality data from file or generate synthetic data for demonstration.
        """
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                print(f"Data loaded successfully from {file_path}")
            except Exception as e:
                print(f"Error loading data: {e}")
                print("Generating synthetic data for demonstration...")
                self.data = self._generate_synthetic_data()
        else:
            print("No file path provided. Generating synthetic data for demonstration...")
            self.data = self._generate_synthetic_data()
        
        return self.data
    
    def _generate_synthetic_data(self, n_samples=5000):
        """
        Generate synthetic air quality data for demonstration purposes.
        """
        np.random.seed(42)
        
        # Generate time-based features
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='H')
        
        # Generate weather features
        temperature = np.random.normal(20, 10, n_samples)  # Temperature in Celsius
        humidity = np.random.uniform(30, 90, n_samples)    # Humidity percentage
        wind_speed = np.random.exponential(5, n_samples)   # Wind speed in km/h
        pressure = np.random.normal(1013, 20, n_samples)   # Atmospheric pressure in hPa
        
        # Generate pollutant concentrations with realistic correlations
        # PM2.5 (target variable)
        pm25_base = 15 + 0.3 * temperature + 0.1 * humidity - 0.5 * wind_speed
        pm25 = np.maximum(0, pm25_base + np.random.normal(0, 5, n_samples))
        
        # PM10 (correlated with PM2.5)
        pm10 = pm25 * 1.5 + np.random.normal(0, 3, n_samples)
        pm10 = np.maximum(0, pm10)
        
        # NO2
        no2_base = 25 + 0.2 * temperature - 0.3 * wind_speed
        no2 = np.maximum(0, no2_base + np.random.normal(0, 8, n_samples))
        
        # SO2
        so2_base = 10 - 0.1 * wind_speed
        so2 = np.maximum(0, so2_base + np.random.normal(0, 3, n_samples))
        
        # CO
        co_base = 1.2 + 0.05 * temperature - 0.02 * wind_speed
        co = np.maximum(0, co_base + np.random.normal(0, 0.3, n_samples))
        
        # O3 (ozone)
        o3_base = 50 + 0.8 * temperature - 0.2 * humidity
        o3 = np.maximum(0, o3_base + np.random.normal(0, 10, n_samples))
        
        # Create time-based features
        hour = dates.hour
        day_of_week = dates.dayofweek
        month = dates.month
        
        # Create seasonal patterns
        seasonal_factor = np.sin(2 * np.pi * dates.dayofyear / 365)
        
        # Create DataFrame
        data = pd.DataFrame({
            'datetime': dates,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pressure': pressure,
            'pm10': pm10,
            'no2': no2,
            'so2': so2,
            'co': co,
            'o3': o3,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'seasonal_factor': seasonal_factor,
            'pm25': pm25  # Target variable
        })
        
        print(f"Generated synthetic dataset with {n_samples} samples and {len(data.columns)} features")
        return data
    
    def explore_data(self):
        """
        Perform exploratory data analysis on the dataset.
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic information
        print("\nDataset Shape:", self.data.shape)
        print("\nColumn Names:", list(self.data.columns))
        print("\nData Types:")
        print(self.data.dtypes)
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.data.describe())
        
        # Missing values
        print("\nMissing Values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Correlation analysis
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_columns].corr()
        
        # Create visualizations
        self._create_eda_plots(correlation_matrix)
        
        return correlation_matrix
    
    def _create_eda_plots(self, correlation_matrix):
        """
        Create exploratory data analysis plots.
        """
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Correlation heatmap
        plt.subplot(2, 3, 1)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 2. PM2.5 distribution
        plt.subplot(2, 3, 2)
        plt.hist(self.data['pm25'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('PM2.5 Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('PM2.5 Concentration (μg/m³)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 3. Time series plot of PM2.5
        plt.subplot(2, 3, 3)
        if 'datetime' in self.data.columns:
            sample_data = self.data.sample(min(1000, len(self.data))).sort_values('datetime')
            plt.plot(sample_data['datetime'], sample_data['pm25'], alpha=0.7, color='red')
            plt.title('PM2.5 Time Series (Sample)', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('PM2.5 Concentration (μg/m³)')
            plt.xticks(rotation=45)
        else:
            plt.plot(self.data['pm25'][:1000], alpha=0.7, color='red')
            plt.title('PM2.5 Time Series (First 1000 points)', fontsize=14, fontweight='bold')
            plt.xlabel('Time Index')
            plt.ylabel('PM2.5 Concentration (μg/m³)')
        plt.grid(True, alpha=0.3)
        
        # 4. Scatter plot: Temperature vs PM2.5
        plt.subplot(2, 3, 4)
        plt.scatter(self.data['temperature'], self.data['pm25'], alpha=0.5, color='green')
        plt.title('Temperature vs PM2.5', fontsize=14, fontweight='bold')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('PM2.5 Concentration (μg/m³)')
        plt.grid(True, alpha=0.3)
        
        # 5. Box plot: PM2.5 by hour of day
        plt.subplot(2, 3, 5)
        if 'hour' in self.data.columns:
            sns.boxplot(x='hour', y='pm25', data=self.data)
            plt.title('PM2.5 by Hour of Day', fontsize=14, fontweight='bold')
            plt.xlabel('Hour of Day')
            plt.ylabel('PM2.5 Concentration (μg/m³)')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'Hour data not available', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('PM2.5 by Hour of Day', fontsize=14, fontweight='bold')
        
        # 6. Feature importance preview (using correlation with target)
        plt.subplot(2, 3, 6)
        target_corr = correlation_matrix['pm25'].abs().sort_values(ascending=True)
        target_corr = target_corr[target_corr.index != 'pm25']  # Remove self-correlation
        target_corr.plot(kind='barh', color='orange')
        plt.title('Feature Correlation with PM2.5', fontsize=14, fontweight='bold')
        plt.xlabel('Absolute Correlation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nExploratory Data Analysis plots saved as 'eda_plots.png'")
    
    def preprocess_data(self, target_column='pm25'):
        """
        Preprocess the data for machine learning.
        """
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Remove datetime column if present
        if 'datetime' in self.data.columns:
            self.data = self.data.drop('datetime', axis=1)
        
        # Separate features and target
        self.target_name = target_column
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        
        self.feature_names = list(X.columns)
        
        print(f"Features: {self.feature_names}")
        print(f"Target: {self.target_name}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Handle missing values
        print("\nHandling missing values...")
        X_imputed = self.imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=self.feature_names)
        
        # Feature scaling
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Split the data
        print("Splitting data into train and test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train multiple machine learning models.
        """
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # Train each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Store the trained model
            self.models[name] = model
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            
            print(f"Cross-validation RMSE: {cv_rmse.mean():.3f} (+/- {cv_rmse.std() * 2:.3f})")
        
        print("\nAll models trained successfully!")
        return self.models
    
    def evaluate_models(self):
        """
        Evaluate all trained models and compare their performance.
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Make predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            
            train_mse = mean_squared_error(self.y_train, y_pred_train)
            test_mse = mean_squared_error(self.y_test, y_pred_test)
            
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            
            # Store results
            results[name] = {
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Train MSE': train_mse,
                'Test MSE': test_mse,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Predictions': y_pred_test
            }
            
            print(f"Train MAE: {train_mae:.3f}, Test MAE: {test_mae:.3f}")
            print(f"Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}")
            print(f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            model: {metric: results[model][metric] for metric in 
                   ['Test MAE', 'Test RMSE', 'Test R²']}
            for model in results.keys()
        }).T
        
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        print(comparison_df.round(3))
        
        # Find best model
        best_model_name = comparison_df['Test R²'].idxmax()
        print(f"\nBest performing model: {best_model_name}")
        print(f"Test R² Score: {comparison_df.loc[best_model_name, 'Test R²']:.3f}")
        
        self.results = results
        self.best_model_name = best_model_name
        
        return results, comparison_df
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations of the results.
        """
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Performance Comparison
        plt.subplot(2, 3, 1)
        models = list(self.results.keys())
        test_r2_scores = [self.results[model]['Test R²'] for model in models]
        test_rmse_scores = [self.results[model]['Test RMSE'] for model in models]
        
        x_pos = np.arange(len(models))
        bars = plt.bar(x_pos, test_r2_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.title('Model Performance Comparison (R² Score)', fontsize=14, fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.xticks(x_pos, models, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. RMSE Comparison
        plt.subplot(2, 3, 2)
        bars = plt.bar(x_pos, test_rmse_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.title('Model Performance Comparison (RMSE)', fontsize=14, fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.xticks(x_pos, models, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Actual vs Predicted (Best Model)
        plt.subplot(2, 3, 3)
        best_predictions = self.results[self.best_model_name]['Predictions']
        plt.scatter(self.y_test, best_predictions, alpha=0.6, color='blue')
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), best_predictions.min())
        max_val = max(self.y_test.max(), best_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.title(f'Actual vs Predicted ({self.best_model_name})', fontsize=14, fontweight='bold')
        plt.xlabel('Actual PM2.5')
        plt.ylabel('Predicted PM2.5')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Residuals Plot (Best Model)
        plt.subplot(2, 3, 4)
        residuals = self.y_test - best_predictions
        plt.scatter(best_predictions, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'Residuals Plot ({self.best_model_name})', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted PM2.5')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        
        # 5. Feature Importance (if available)
        plt.subplot(2, 3, 5)
        best_model = self.models[self.best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='orange')
            plt.title(f'Feature Importance ({self.best_model_name})', fontsize=14, fontweight='bold')
            plt.xlabel('Importance')
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available for\nthis model type', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'Feature Importance ({self.best_model_name})', fontsize=14, fontweight='bold')
        
        # 6. Error Distribution
        plt.subplot(2, 3, 6)
        plt.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Error Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Model results visualization saved as 'model_results.png'")
        
        # Create additional detailed plots
        self._create_detailed_analysis_plots()
    
    def _create_detailed_analysis_plots(self):
        """
        Create additional detailed analysis plots.
        """
        # Create a separate figure for detailed analysis
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Learning Curves (for Random Forest)
        plt.subplot(2, 2, 1)
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_scores = []
            val_scores = []
            
            for train_size in train_sizes:
                # Sample training data
                sample_size = int(train_size * len(self.X_train))
                X_sample = self.X_train[:sample_size]
                y_sample = self.y_train[:sample_size]
                
                # Train model on sample
                rf_temp = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_temp.fit(X_sample, y_sample)
                
                # Evaluate
                train_pred = rf_temp.predict(X_sample)
                val_pred = rf_temp.predict(self.X_test)
                
                train_scores.append(r2_score(y_sample, train_pred))
                val_scores.append(r2_score(self.y_test, val_pred))
            
            plt.plot(train_sizes, train_scores, 'o-', color='blue', label='Training Score')
            plt.plot(train_sizes, val_scores, 'o-', color='red', label='Validation Score')
            plt.title('Learning Curves (Random Forest)', fontsize=14, fontweight='bold')
            plt.xlabel('Training Set Size')
            plt.ylabel('R² Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Prediction Intervals
        plt.subplot(2, 2, 2)
        best_predictions = self.results[self.best_model_name]['Predictions']
        sorted_indices = np.argsort(self.y_test)
        sorted_actual = self.y_test.iloc[sorted_indices]
        sorted_predicted = best_predictions[sorted_indices]
        
        plt.plot(range(len(sorted_actual)), sorted_actual, 'b-', alpha=0.7, label='Actual')
        plt.plot(range(len(sorted_predicted)), sorted_predicted, 'r-', alpha=0.7, label='Predicted')
        plt.fill_between(range(len(sorted_predicted)), 
                        sorted_predicted - 2, sorted_predicted + 2, 
                        alpha=0.3, color='red', label='±2 μg/m³')
        plt.title('Prediction Intervals', fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index (sorted by actual value)')
        plt.ylabel('PM2.5 Concentration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Model Complexity Analysis
        plt.subplot(2, 2, 3)
        if 'Random Forest' in self.models:
            n_estimators_range = [10, 25, 50, 100, 150, 200]
            train_scores = []
            test_scores = []
            
            for n_est in n_estimators_range:
                rf_temp = RandomForestRegressor(n_estimators=n_est, random_state=42)
                rf_temp.fit(self.X_train, self.y_train)
                
                train_pred = rf_temp.predict(self.X_train)
                test_pred = rf_temp.predict(self.X_test)
                
                train_scores.append(r2_score(self.y_train, train_pred))
                test_scores.append(r2_score(self.y_test, test_pred))
            
            plt.plot(n_estimators_range, train_scores, 'o-', color='blue', label='Training Score')
            plt.plot(n_estimators_range, test_scores, 'o-', color='red', label='Test Score')
            plt.title('Model Complexity Analysis (Random Forest)', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Estimators')
            plt.ylabel('R² Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. Cross-Validation Scores
        plt.subplot(2, 2, 4)
        cv_results = {}
        for name, model in self.models.items():
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                      cv=5, scoring='r2')
            cv_results[name] = cv_scores
        
        # Create box plot
        plt.boxplot([cv_results[name] for name in cv_results.keys()], 
                   labels=list(cv_results.keys()))
        plt.title('Cross-Validation R² Scores', fontsize=14, fontweight='bold')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Detailed analysis plots saved as 'detailed_analysis.png'")
    
    def generate_predictions(self, new_data=None):
        """
        Generate predictions using the best model.
        """
        if new_data is None:
            # Use test data for demonstration
            predictions = self.results[self.best_model_name]['Predictions']
            actual = self.y_test
        else:
            # Preprocess new data
            new_data_scaled = self.scaler.transform(new_data)
            predictions = self.models[self.best_model_name].predict(new_data_scaled)
            actual = None
        
        return predictions, actual
    
    def save_results(self):
        """
        Save model results and predictions to files.
        """
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)
        
        # Save model comparison
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Test_MAE': results['Test MAE'],
                'Test_RMSE': results['Test RMSE'],
                'Test_R2': results['Test R²']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('model_comparison.csv', index=False)
        print("Model comparison saved to 'model_comparison.csv'")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'Actual': self.y_test,
            'Predicted': self.results[self.best_model_name]['Predictions'],
            'Error': self.y_test - self.results[self.best_model_name]['Predictions']
        })
        predictions_df.to_csv('predictions.csv', index=False)
        print("Predictions saved to 'predictions.csv'")
        
        # Save feature importance (if available)
        best_model = self.models[self.best_model_name]
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            importance_df.to_csv('feature_importance.csv', index=False)
            print("Feature importance saved to 'feature_importance.csv'")
        
        print("All results saved successfully!")

def main():
    """
    Main function to run the air quality prediction system.
    """
    print("="*60)
    print("AIR QUALITY PREDICTION SYSTEM USING MACHINE LEARNING")
    print("="*60)
    
    # Initialize the predictor
    predictor = AirQualityPredictor()
    
    # Load data (using synthetic data for demonstration)
    data = predictor.load_data()
    
    # Explore the data
    correlation_matrix = predictor.explore_data()
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = predictor.preprocess_data()
    
    # Train models
    models = predictor.train_models()
    
    # Evaluate models
    results, comparison_df = predictor.evaluate_models()
    
    # Create visualizations
    predictor.create_visualizations()
    
    # Save results
    predictor.save_results()
    
    print("\n" + "="*60)
    print("AIR QUALITY PREDICTION SYSTEM COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Best Model: {predictor.best_model_name}")
    print(f"Test R² Score: {results[predictor.best_model_name]['Test R²']:.3f}")
    print(f"Test RMSE: {results[predictor.best_model_name]['Test RMSE']:.3f}")
    print("\nGenerated Files:")
    print("- eda_plots.png: Exploratory data analysis visualizations")
    print("- model_results.png: Model performance comparison and results")
    print("- detailed_analysis.png: Detailed analysis plots")
    print("- model_comparison.csv: Model performance metrics")
    print("- predictions.csv: Actual vs predicted values")
    print("- feature_importance.csv: Feature importance rankings")

if __name__ == "__main__":
    main()

