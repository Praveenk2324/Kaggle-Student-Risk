
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Import the preprocessing function you just created
from preprocess import preprocess_data

def get_next_filename(base_dir, base_name="submission", ext=".csv"):
    """
    Finds the next available filename to avoid overwriting old submissions.
    Example: submission_1.csv, submission_2.csv
    """
    counter = 1
    while True:
        filename = os.path.join(base_dir, f"{base_name}_{counter}{ext}")
        if not os.path.exists(filename):
            return filename
        counter += 1

def main():
    # Setup paths relative to this script's location
    # This ensures it works no matter where you run it from
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(src_dir, '..'))
    
    data_dir = os.path.join(project_dir, 'data', 'raw')
    results_dir = os.path.join(project_dir, 'results')
    
    # Create the results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load data
    print("Loading training data...")
    train_path = os.path.join(data_dir, 'train.csv')
    train_data = pd.read_csv(train_path)
    
    # 2. Preprocess
    print("Preprocessing training data...")
    df = preprocess_data(train_data)
    
    # ==========================================
    # PHASE 1: LOCAL VALIDATION
    # ==========================================
    print("\n--- PHASE 1: Local Validation ---")
    X = df.drop('health_condition', axis=1)
    y = df['health_condition']
    
    # Split into Local Train/Test
    X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    local_model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    print("Training local model...")
    local_model.fit(X_train_local, y_train_local)
    
    local_preds = local_model.predict(X_test_local)
    print("Local Validation Score:\n")
    print(classification_report(y_test_local, local_preds))
    
    # ==========================================
    # PHASE 2: FINAL MODEL & SUBMISSION
    # ==========================================
    print("\n--- PHASE 2: Final Predictions ---")
    # Retrain on 100% of the Training Data
    final_model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    print("Retraining model on ALL training data...")
    final_model.fit(X, y)
    
    print("Loading and preprocessing test data...")
    test_path = os.path.join(data_dir, 'test.csv')
    raw_test_df = pd.read_csv(test_path)
    passenger_ids = raw_test_df['id']
    
    # Preprocess test data
    clean_test_df = preprocess_data(raw_test_df)
    
    # Align columns perfectly with training data
    X_test_kaggle = clean_test_df.reindex(columns=X.columns, fill_value=0)
    
    print("Generating predictions...")
    kaggle_preds = final_model.predict(X_test_kaggle)
    
    # Format submission
    submission = pd.DataFrame({
        'id': passenger_ids,
        'health_condition': kaggle_preds
    })
    
    # Map numeric predictions (0, 1, 2) back to text labels for Kaggle
    reverse_mapping = {0: 'at-risk', 1: 'fit', 2: 'unhealthy'}
    submission['health_condition'] = submission['health_condition'].map(reverse_mapping)
    
    # Get the next incremental filename and save
    save_path = get_next_filename(results_dir)
    submission.to_csv(save_path, index=False)
    print(f"\nSuccess! Submission saved at: {save_path}")

if __name__ == "__main__":
    main()


