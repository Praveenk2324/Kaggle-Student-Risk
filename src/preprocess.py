import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the dataset according to the steps defined in eda.ipynb.
    Handles missing values, encoding, and data types.
    This function is designed to be reusable for both training and new datasets.
    """
    df = df.copy()
    
    # 1. Health Condition Label Encoding (only applies to Training data)
    if 'health_condition' in df.columns and df['health_condition'].dtype == 'object':
        # Using map instead of LabelEncoder to ensure consistent 0, 1, 2 mapping
        health_mapping = {'at-risk': 0, 'fit': 1, 'unhealthy': 2}
        df['health_condition'] = df['health_condition'].map(health_mapping)
        
    # 2. Diet Type: drop nulls and one-hot encode
    # 2. Diet Type: impute missing values to preserve all rows, then one-hot encode
    if 'diet_type' in df.columns:
        # Fill missing diet types with the most frequent value (mode) instead of dropping
        df['diet_type'] = df['diet_type'].fillna(df['diet_type'].mode()[0] if not df['diet_type'].mode().empty else 'Unknown')
        df = pd.get_dummies(df, columns=['diet_type'], drop_first=True, dtype=int)
        
    # 3. Stress Level: Impute using global mode to prevent target leakage
    if 'stress_level' in df.columns:
        # Replaced health_condition groupby with global mode
        df['stress_level'] = df['stress_level'].fillna(df['stress_level'].mode()[0] if not df['stress_level'].mode().empty else 'medium')
            
        stress_mapping = {'low': 0, 'medium': 1, 'high': 2}
        if df['stress_level'].dtype == 'object':
            df['stress_level'] = df['stress_level'].map(stress_mapping)
            
    # 4. Sleep Quality: impute based on stress_level mode, then map
    if 'sleep_quality' in df.columns:
        if 'stress_level' in df.columns:
            df['sleep_quality'] = df.groupby('stress_level')['sleep_quality'].transform(
                lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'average')
            )
        else:
            df['sleep_quality'] = df['sleep_quality'].fillna(df['sleep_quality'].mode()[0] if not df['sleep_quality'].mode().empty else 'average')
            
        sleep_mapping = {'poor': 0, 'average': 1, 'good': 2}
        if df['sleep_quality'].dtype == 'object':
            df['sleep_quality'] = df['sleep_quality'].map(sleep_mapping)

    # 5. Physical Activity Level: impute based on exercise_duration quantiles, then map
    if 'physical_activity_level' in df.columns:
        if 'exercise_duration' in df.columns:
            try:
                duration_buckets = pd.qcut(df['exercise_duration'], q=4, duplicates='drop')
                df['physical_activity_level'] = df.groupby(duration_buckets, observed=False)['physical_activity_level'].transform(
                    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'moderate')
                )
            except ValueError:
                df['physical_activity_level'] = df['physical_activity_level'].fillna(df['physical_activity_level'].mode()[0] if not df['physical_activity_level'].mode().empty else 'moderate')
        else:
            df['physical_activity_level'] = df['physical_activity_level'].fillna(df['physical_activity_level'].mode()[0] if not df['physical_activity_level'].mode().empty else 'moderate')
            
        physical_mapping = {'sedentary': 0, 'moderate': 1, 'active': 2}
        if df['physical_activity_level'].dtype == 'object':
            df['physical_activity_level'] = df['physical_activity_level'].map(physical_mapping)
            
        # Fill any remaining NaNs with 1 (moderate) and cast to int
        df['physical_activity_level'] = df['physical_activity_level'].fillna(1).astype(int)

    # 6. Smoking/Alcohol: Impute using global mode to prevent target leakage
    if 'smoking_alcohol' in df.columns:
        # Replaced health_condition groupby with global mode
        df['smoking_alcohol'] = df['smoking_alcohol'].fillna(df['smoking_alcohol'].mode()[0] if not df['smoking_alcohol'].mode().empty else 'yes')
            
        habit_mapping = {'no': 0, 'occasional': 1, 'yes': 2}
        if df['smoking_alcohol'].dtype == 'object':
            df['smoking_alcohol'] = df['smoking_alcohol'].map(habit_mapping)

    # 7. Gender: impute with mode, one-hot encode
    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna(df['gender'].mode()[0] if not df['gender'].mode().empty else 'other')
        df = pd.get_dummies(df, columns=['gender'], drop_first=True, dtype=int)

    # 8. Drop ID
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # 9. Sleep Duration: impute based on sleep_quality median
    if 'sleep_duration' in df.columns:
        if 'sleep_quality' in df.columns:
            df['sleep_duration'] = df.groupby('sleep_quality')['sleep_duration'].transform(
                lambda x: x.fillna(x.median() if not x.isna().all() else np.nan)
            )
        # Catch leftover NaNs with global median
        df['sleep_duration'] = df['sleep_duration'].fillna(df['sleep_duration'].median())

    # 10. Other Numerics: impute based on physical_activity_level median
    numeric_cols = ['heart_rate', 'bmi', 'calorie_expenditure', 'step_count', 'exercise_duration', 'water_intake']
    for col in numeric_cols:
        if col in df.columns:
            if 'physical_activity_level' in df.columns:
                df[col] = df.groupby('physical_activity_level')[col].transform(
                    lambda x: x.fillna(x.median() if not x.isna().all() else np.nan)
                )
            # Catch leftover NaNs with global median
            df[col] = df[col].fillna(df[col].median())

    # Optional: ensure consistency of dummy columns if expected
    expected_dummies = ['diet_type_non-veg', 'diet_type_veg', 'gender_male', 'gender_other']
    for col in expected_dummies:
        if col not in df.columns and any(base in col for base in ['diet_type', 'gender']):
            # Just create missing expected dummy columns with 0
            df[col] = 0

    return df