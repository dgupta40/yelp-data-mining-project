#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re
import os
import csv
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

def load_hygiene_dataset(data_dir):
    print("Loading hygiene dataset...")
    
    # Load review texts
    review_texts = []
    with open(os.path.join(data_dir, 'hygiene.dat'), 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            review_texts.append(line.strip())
    
    # Load inspection labels
    inspection_labels = []
    with open(os.path.join(data_dir, 'hygiene.dat.labels'), 'r', encoding='utf-8') as f:
        for line in f:
            label = line.strip()
            if label == '[None]':
                inspection_labels.append(None)
            else:
                inspection_labels.append(int(label))
    
    # Load restaurant metadata
    restaurant_metadata = []
    with open(os.path.join(data_dir, 'hygiene.dat.additional'), 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            try:
                cuisines = row[0] if len(row) > 0 else ''
                zipcode = row[1] if len(row) > 1 else ''
                num_reviews = int(row[2]) if len(row) > 2 and row[2].strip().isdigit() else 0
                avg_rating = float(row[3]) if len(row) > 3 and row[3].strip() else 0.0
                
                restaurant_metadata.append([cuisines, zipcode, num_reviews, avg_rating])
                
            except Exception:
                restaurant_metadata.append(['', '', 0, 0.0])
    
    # Create dataset DataFrame
    dataset = pd.DataFrame({
        'review_text': review_texts,
        'inspection_result': inspection_labels,
        'cuisines': [d[0] for d in restaurant_metadata],
        'zipcode': [d[1] for d in restaurant_metadata],
        'num_reviews': [d[2] for d in restaurant_metadata],
        'avg_rating': [d[3] for d in restaurant_metadata]
    })
    
    print(f"Loaded {len(dataset)} restaurants")
    
    # Split into training and test sets
    train_set = dataset[dataset['inspection_result'].notna()].copy()
    test_set = dataset[dataset['inspection_result'].isna()].copy()
    
    print(f"Training set: {len(train_set)} restaurants")
    print(f"Test set: {len(test_set)} restaurants")
    print(f"Training label distribution: {train_set['inspection_result'].value_counts().to_dict()}")
    
    return train_set, test_set

def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def calculate_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

def create_text_features(train_texts, test_texts):
    print("Creating text features...")
    
    # Text preprocessing
    train_processed = [preprocess_text(t) for t in train_texts]
    test_processed = [preprocess_text(t) for t in test_texts]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    train_tfidf = vectorizer.fit_transform(train_processed)
    test_tfidf = vectorizer.transform(test_processed)
    
    # Sentiment analysis
    train_sentiment = [calculate_sentiment(t) for t in train_processed]
    test_sentiment = [calculate_sentiment(t) for t in test_processed]
    
    # Text length features
    train_length = [len(t.split()) for t in train_processed]
    test_length = [len(t.split()) for t in test_processed]
    
    print(f"TF-IDF features: {train_tfidf.shape[1]}")
    
    return train_tfidf, test_tfidf, train_sentiment, test_sentiment, train_length, test_length

def create_cuisine_features(train_cuisines, test_cuisines):
    print("Creating cuisine features...")
    
    def parse_cuisine_list(cuisine_str):
        try:
            if pd.isna(cuisine_str) or cuisine_str == '':
                return []
            
            cuisine_str = str(cuisine_str).strip()
            if cuisine_str.startswith('[') and cuisine_str.endswith(']'):
                cuisine_str = cuisine_str[1:-1]
            
            cuisines = [c.strip().strip("'\"") for c in cuisine_str.split(',')]
            return [c for c in cuisines if c and c.lower() != 'restaurants']
        except:
            return []
    
    train_cuisine_lists = [parse_cuisine_list(c) for c in train_cuisines]
    test_cuisine_lists = [parse_cuisine_list(c) for c in test_cuisines]
    
    # Build cuisine vocabulary
    all_cuisines = set()
    for cuisine_list in train_cuisine_lists + test_cuisine_lists:
        all_cuisines.update(cuisine_list)
    
    cuisine_vocab = sorted(list(all_cuisines))[:50]
    print(f"Using {len(cuisine_vocab)} cuisine types")
    
    def encode_cuisines(cuisine_lists):
        features = []
        for cuisine_list in cuisine_lists:
            feature_vector = [1 if cuisine in cuisine_list else 0 for cuisine in cuisine_vocab]
            feature_vector.append(len(cuisine_list))  # Total cuisine count
            features.append(feature_vector)
        return np.array(features)
    
    train_features = encode_cuisines(train_cuisine_lists)
    test_features = encode_cuisines(test_cuisine_lists)
    
    return train_features, test_features

def create_location_features(train_zipcodes, test_zipcodes):
    print("Creating location features...")
    
    unique_zipcodes = list(set(list(train_zipcodes) + list(test_zipcodes)))
    zipcode_mapping = {zipcode: i for i, zipcode in enumerate(unique_zipcodes)}
    
    train_encoded = [zipcode_mapping.get(z, 0) for z in train_zipcodes]
    test_encoded = [zipcode_mapping.get(z, 0) for z in test_zipcodes]
    
    return np.array(train_encoded).reshape(-1, 1), np.array(test_encoded).reshape(-1, 1)

def combine_feature_matrices(tfidf_train, tfidf_test, 
                           sentiment_train, sentiment_test,
                           length_train, length_test,
                           cuisine_train, cuisine_test,
                           location_train, location_test,
                           numeric_train, numeric_test):
    print("Combining feature matrices...")
    
    tfidf_train_dense = tfidf_train.toarray()
    tfidf_test_dense = tfidf_test.toarray()
    
    X_train = np.hstack([
        tfidf_train_dense,
        np.array(sentiment_train).reshape(-1, 1),
        np.array(length_train).reshape(-1, 1),
        cuisine_train,
        location_train,
        numeric_train
    ])
    
    X_test = np.hstack([
        tfidf_test_dense,
        np.array(sentiment_test).reshape(-1, 1),
        np.array(length_test).reshape(-1, 1),
        cuisine_test,
        location_test,
        numeric_test
    ])
    
    print(f"Final feature matrix shape: {X_train.shape}")
    
    return X_train, X_test

def train_classification_models(X_train, y_train):
    print("Training classification models...")
    
    models = {}
    
    # Support Vector Machine
    print("Training SVM...")
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train, y_train)
    models['SVM'] = svm_model
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    
    return models

def evaluate_models(models, X_train, y_train):
    print("Evaluating models with cross-validation...")
    
    cv_results = {}
    for model_name, model in models.items():
        f1_scores = cross_val_score(model, X_train, y_train, 
                                   cv=5, scoring='f1_macro')
        cv_results[model_name] = f1_scores.mean()
        print(f"{model_name}: F1 = {f1_scores.mean():.3f}")
    
    return cv_results

def optimize_threshold(model, X_train, y_train):
    print("Optimizing classification threshold...")
    
    prediction_probabilities = model.predict_proba(X_train)[:, 1]
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        predictions = (prediction_probabilities >= threshold).astype(int)
        f1 = f1_score(y_train, predictions, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    return best_threshold

def create_diagnostic_plots(train_data, model_results, save_dir):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Label distribution
    train_data['inspection_result'].value_counts().plot(kind='bar', ax=ax1, color=['green', 'red'])
    ax1.set_title('Inspection Results Distribution')
    ax1.set_xlabel('0=Pass, 1=Fail')
    ax1.set_ylabel('Count')
    
    # Rating distribution by label
    for label in [0, 1]:
        subset = train_data[train_data['inspection_result'] == label]
        ax2.hist(subset['avg_rating'], alpha=0.5, label=f'Label {label}', bins=20)
    ax2.set_title('Rating Distribution by Inspection Result')
    ax2.set_xlabel('Average Rating')
    ax2.legend()
    
    # Model performance comparison
    model_names = list(model_results.keys())
    f1_scores = [model_results[m] for m in model_names]
    ax3.bar(model_names, f1_scores)
    ax3.set_title('Cross-Validation F1 Scores')
    ax3.set_ylabel('F1 Score')
    ax3.tick_params(axis='x', rotation=45)
    
    # Text length distribution
    train_data['text_length'] = train_data['review_text'].str.len()
    for label in [0, 1]:
        subset = train_data[train_data['inspection_result'] == label]
        ax4.hist(subset['text_length'], alpha=0.5, label=f'Label {label}', bins=30)
    ax4.set_title('Review Text Length Distribution')
    ax4.set_xlabel('Character Count')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'task6_plots.png'), dpi=300, bbox_inches='tight')
    plt.show()

def save_predictions(predictions, save_dir):
    output_file = os.path.join(save_dir, 'predictions.txt')
    
    with open(output_file, 'w') as f:
        f.write("student_hygiene\n")
        for pred in predictions:
            f.write(f"{pred}\n")
    
    print(f"Predictions saved to: {output_file}")
    return output_file

def main():
    data_dir = r'C:\Users\dheer\OneDrive\Desktop\Coursera Data Mining\Hygiene'
    save_dir = r'C:\Users\dheer\OneDrive\Desktop\Coursera Data Mining\Visuals'
    os.makedirs(save_dir, exist_ok=True)
    
    print("Task 6: Predicting Hygiene Failures")
    print("=" * 35)
    
    # Load dataset
    train_data, test_data = load_hygiene_dataset(data_dir)
    
    # Feature engineering
    print("\nFeature Engineering:")
    
    # Text features
    train_tfidf, test_tfidf, train_sentiment, test_sentiment, train_length, test_length = create_text_features(
        train_data['review_text'], test_data['review_text'])
    
    # Cuisine features
    train_cuisine, test_cuisine = create_cuisine_features(
        train_data['cuisines'], test_data['cuisines'])
    
    # Location features
    train_location, test_location = create_location_features(
        train_data['zipcode'], test_data['zipcode'])
    
    # Numeric features (scaled)
    train_numeric = train_data[['num_reviews', 'avg_rating']].values
    test_numeric = test_data[['num_reviews', 'avg_rating']].values
    
    scaler = StandardScaler()
    train_numeric_scaled = scaler.fit_transform(train_numeric)
    test_numeric_scaled = scaler.transform(test_numeric)
    
    # Combine all features
    X_train, X_test = combine_feature_matrices(
        train_tfidf, test_tfidf,
        train_sentiment, test_sentiment,
        train_length, test_length,
        train_cuisine, test_cuisine,
        train_location, test_location,
        train_numeric_scaled, test_numeric_scaled
    )
    
    y_train = train_data['inspection_result'].values
    
    # Model training and evaluation
    models = train_classification_models(X_train, y_train)
    model_results = evaluate_models(models, X_train, y_train)
    
    # Select best model
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k])
    best_model = models[best_model_name]
    print(f"\nBest performing model: {best_model_name}")
    
    # Threshold optimization
    optimal_threshold = optimize_threshold(best_model, X_train, y_train)
    
    # Final predictions
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    final_predictions = (test_probabilities >= optimal_threshold).astype(int)
    
    print(f"\nPrediction Results:")
    print(f"Predicted failures: {sum(final_predictions)} / {len(final_predictions)}")
    print(f"Predicted failure rate: {sum(final_predictions) / len(final_predictions):.3f}")
    
    # Save results
    prediction_file = save_predictions(final_predictions, save_dir)
    create_diagnostic_plots(train_data, model_results, save_dir)
    
    # Save summary report
    summary_file = os.path.join(save_dir, 'task6_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Task 6: Hygiene Failure Prediction Results\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"Training instances: {len(train_data)}\n")
        f.write(f"Test instances: {len(test_data)}\n")
        f.write(f"Feature dimensions: {X_train.shape[1]}\n\n")
        
        f.write("Model Performance (5-fold CV):\n")
        for model_name, score in model_results.items():
            f.write(f"{model_name}: {score:.3f}\n")
        
        f.write(f"\nBest Model: {best_model_name}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.3f}\n")
        f.write(f"Predicted Failures: {sum(final_predictions)}\n")
        f.write(f"Failure Rate: {sum(final_predictions) / len(final_predictions):.3f}\n")
    
    print(f"\nAnalysis complete! Results saved to {save_dir}")

if __name__ == "__main__":
    main()