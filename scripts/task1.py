#!/usr/bin/env python3

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# NLTK setup
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def load_reviews(file_path, sample_size=50000):
    reviews = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            reviews.append(json.loads(line))
    return pd.DataFrame(reviews)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [lemmatizer.lemmatize(w) for w in words
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def extract_topics(lda_model, feature_names, n_words=10):
    topics = []
    for i, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        topics.append(f"Topic {i + 1}: {', '.join(top_words)}")
    return topics

def main():
    base_dir = r'C:\Users\dheer\OneDrive\Desktop\Coursera Data Mining\yelp_dataset_challenge_academic_dataset'
    review_file = os.path.join(base_dir, 'yelp_academic_dataset_review.json')
    save_dir = r'C:\Users\dheer\OneDrive\Desktop\Coursera Data Mining\Visuals'
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading reviews...")
    reviews_df = load_reviews(review_file, sample_size=50000)
    print(f"Loaded {len(reviews_df)} reviews")
    
    # Star rating distribution
    plt.figure(figsize=(8, 6))
    reviews_df['stars'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Star Ratings')
    plt.xlabel('Stars')
    plt.ylabel('Number of Reviews')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'star_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("Preprocessing text...")
    reviews_df['processed_text'] = reviews_df['text'].apply(preprocess_text)
    clean_df = reviews_df[reviews_df['processed_text'].str.len() > 0].copy()
    print(f"After preprocessing: {len(clean_df)} reviews")
    
    # Task 1.1: All reviews analysis
    print("\nTask 1.1: Analyzing all reviews")
    
    vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.7)
    doc_term_matrix = vectorizer.fit_transform(clean_df['processed_text'])
    
    lda_model = LatentDirichletAllocation(n_components=10, random_state=42, max_iter=10)
    lda_model.fit(doc_term_matrix)
    
    features = vectorizer.get_feature_names_out()
    all_topics = extract_topics(lda_model, features)
    
    print("Topics discovered:")
    for topic in all_topics:
        print(topic)
    
    # Plot topic weights
    weights = lda_model.components_.mean(axis=1)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(10), weights, color='lightcoral', alpha=0.7)
    plt.title('Topic Weights - All Reviews', fontsize=14)
    plt.xlabel('Topic Number')
    plt.ylabel('Average Weight')
    plt.xticks(range(10), [f'Topic {i+1}' for i in range(10)], rotation=45)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_topics.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Task 1.2: Positive vs negative comparison
    print("\nTask 1.2: Comparing positive vs negative reviews")
    
    positive_reviews = clean_df[clean_df['stars'] >= 4].sample(
        n=min(10000, len(clean_df[clean_df['stars'] >= 4])), random_state=42)
    negative_reviews = clean_df[clean_df['stars'] <= 2].sample(
        n=min(10000, len(clean_df[clean_df['stars'] <= 2])), random_state=42)
    
    print(f"Positive: {len(positive_reviews)}, Negative: {len(negative_reviews)}")
    
    # Separate vectorizers for pos/neg
    pos_vectorizer = CountVectorizer(max_features=1000, min_df=3, max_df=0.7)
    neg_vectorizer = CountVectorizer(max_features=1000, min_df=3, max_df=0.7)
    
    pos_matrix = pos_vectorizer.fit_transform(positive_reviews['processed_text'])
    neg_matrix = neg_vectorizer.fit_transform(negative_reviews['processed_text'])
    
    pos_lda = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=10)
    neg_lda = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=10)
    
    pos_lda.fit(pos_matrix)
    neg_lda.fit(neg_matrix)
    
    pos_features = pos_vectorizer.get_feature_names_out()
    neg_features = neg_vectorizer.get_feature_names_out()
    
    pos_topics = extract_topics(pos_lda, pos_features)
    neg_topics = extract_topics(neg_lda, neg_features)
    
    print("\nPositive Review Topics:")
    for topic in pos_topics:
        print(topic)
    
    print("\nNegative Review Topics:")
    for topic in neg_topics:
        print(topic)
    
    # Comparison plot
    pos_weights = pos_lda.components_.mean(axis=1)
    neg_weights = neg_lda.components_.mean(axis=1)
    
    x = np.arange(8)
    width = 0.35
    
    plt.figure(figsize=(12, 8))
    bars1 = plt.bar(x - width/2, pos_weights, width, label='Positive Reviews', 
                    alpha=0.8, color='green')
    bars2 = plt.bar(x + width/2, neg_weights, width, label='Negative Reviews', 
                    alpha=0.8, color='red')
    
    plt.xlabel('Topic Number')
    plt.ylabel('Average Weight')
    plt.title('Topic Weights: Positive vs Negative Reviews', fontsize=14)
    plt.xticks(x, [f'Topic {i+1}' for i in range(8)])
    plt.legend()
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pos_vs_neg_topics.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Save results
    results_file = os.path.join(base_dir, 'task1_results.txt')
    with open(results_file, 'w') as f:
        f.write("Task 1: Topic Analysis Results\n")
        f.write("=" * 30 + "\n\n")
        
        f.write("Task 1.1: All Reviews Topics\n")
        f.write("-" * 25 + "\n")
        for topic in all_topics:
            f.write(topic + "\n")
        
        f.write(f"\nTask 1.2: Positive vs Negative\n")
        f.write("-" * 30 + "\n")
        f.write(f"Positive reviews: {len(positive_reviews)}\n")
        f.write(f"Negative reviews: {len(negative_reviews)}\n\n")
        
        f.write("Positive Topics:\n")
        for topic in pos_topics:
            f.write(topic + "\n")
        
        f.write("\nNegative Topics:\n")
        for topic in neg_topics:
            f.write(topic + "\n")
    
    print(f"Results saved to: {results_file}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()