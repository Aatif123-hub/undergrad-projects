# Import required libraries
!pip install -q pandas numpy scikit-learn seaborn matplotlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization in Colab
plt.style.use('seaborn')
sns.set_palette("husl")

class EmployeeProductivityPredictor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None

    def _convert_ratings(self, value):
        """
        Convert text-based ratings to numerical values
        """
        if isinstance(value, (int, float)):
            return value

        rating_map = {
            'very low': 1, 'low': 2, 'medium low': 3, 'medium': 5,
            'medium high': 7, 'high': 8, 'very high': 10,
            'poor': 2, 'fair': 4, 'good': 6, 'very good': 8, 'excellent': 10
        }

        if isinstance(value, str):
            value = value.lower().strip()
            return rating_map.get(value, value)
        return value

    def _preprocess_column(self, df, column):
        """
        Preprocess individual columns based on their content
        """
        # First, handle missing values
        if df[column].isnull().any():
            if df[column].dtype in ['int64', 'float64']:
                df[column].fillna(df[column].median(), inplace=True)
            else:
                df[column].fillna(df[column].mode()[0], inplace=True)

        # Try to convert ratings to numerical values
        try:
            df[column] = df[column].apply(self._convert_ratings)
        except:
            pass

        return df[column]

    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the employee dataset
        """
        print("Loading data...")
        df = pd.read_csv(file_path)

        # Display basic information about the dataset
        print("\nDataset Info:")
        print(f"Total number of records: {len(df)}")
        print("\nColumns in dataset:", df.columns.tolist())
        print("\nSample of raw data:")
        print(df.head())
        print("\nMissing values:")
        print(df.isnull().sum())

        # Identify columns types based on content
        self.categorical_cols = []
        self.numerical_cols = []

        for column in df.columns:
            if column in ['Employee_ID', 'Productivity_Change']:
                continue

            # Preprocess the column
            df[column] = self._preprocess_column(df, column)

            # Check if column should be treated as categorical
            unique_values = df[column].nunique()
            if df[column].dtype == 'object' or unique_values < 10:
                self.categorical_cols.append(column)
            else:
                self.numerical_cols.append(column)

        print("\nCategorical columns:", self.categorical_cols)
        print("Numerical columns:", self.numerical_cols)

        # Convert Productivity_Change to binary
        df['Productivity_Change'] = df['Productivity_Change'].map(
            lambda x: 1 if str(x).lower() in ['increase', 'improved', 'better', '1', 'high'] else 0
        )

        # Encode categorical variables
        print("\nEncoding categorical variables...")
        for col in self.categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))

        # Scale numerical variables
        print("Scaling numerical variables...")
        if self.numerical_cols:  # Only scale if numerical columns exist
            df[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])

        # Create feature matrix X and target variable y
        X = df.drop(['Employee_ID', 'Productivity_Change'] if 'Employee_ID' in df.columns
                    else ['Productivity_Change'], axis=1)
        y = df['Productivity_Change']

        return X, y

    def train_model(self, X, y):
        """
        Train a Random Forest model and evaluate its performance
        """
        print("\nSplitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        print("\nModel Performance:")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_feature_importance(X)

        return X_test, y_test

    def _plot_confusion_matrix(self, y_test, y_pred):
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Decrease', 'Increase'],
            yticklabels=['Decrease', 'Increase']
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def _plot_feature_importance(self, X):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        })
        feature_importance = feature_importance.sort_values(
            'importance', ascending=False
        )

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# Example usage in Colab
def main():
    # Initialize predictor
    predictor = EmployeeProductivityPredictor()

    # Load and preprocess data
    file_path = '/content/mental health.csv'



    try:
        X, y = predictor.load_and_preprocess_data(file_path)

        # Train and evaluate model
        X_test, y_test = predictor.train_model(X, y)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nPlease check your data format and try again.")
        # Print the first few rows of the data for debugging
        try:
            df = pd.read_csv(file_path)
            print("\nFirst few rows of your data:")
            print(df.head())
            print("\nData types of columns:")
            print(df.dtypes)
        except:
            print("Could not read the data file. Please check the file path and format.")

if __name__ == "__main__":
    main()
