#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re
import json
import os
from collections import Counter
import matplotlib.pyplot as plt

def load_labels(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    phrase = parts[0].strip()
                    try:
                        label = int(parts[1].strip())
                        data.append((phrase, label))
                    except ValueError:
                        continue
    
    df = pd.DataFrame(data, columns=['phrase', 'label'])
    df.set_index('phrase', inplace=True)
    return df

def clean_dish_labels(df):
    print("Cleaning labels...")
    
    df_clean = df.copy()
    changes = 0
    
    # Remove non-food items
    non_food = [
        'las vegas', 'san francisco', 'bay area', 'strip mall',
        'credit card', 'belly dancing', 'belly dancer', 'date night',
        'mexican food', 'chinese food', 'middle eastern',
        'salad bar', 'food court', 'main course', 'comfort food', 'fast food'
    ]
    
    for item in non_food:
        if item in df_clean.index:
            df_clean.drop(item, inplace=True)
            changes += 1
    
    # Fix restaurant names labeled as dishes
    restaurant_names = [
        'taj mahal', 'mother india', 'mount everest', 'mt everest', 'india gate',
        'south indian cuisine', 'indian cuisine', 'south asian', 'south indian',
        'great taste', 'hot sauce', 'iced tea', 'ice cream', 'deep fried'
    ]
    
    for item in restaurant_names:
        if item in df_clean.index and df_clean.loc[item, 'label'] == 1:
            df_clean.loc[item, 'label'] = 0
            changes += 1
    
    # Fix mislabeled dishes
    if 'tikka masala' in df_clean.index and df_clean.loc['tikka masala', 'label'] == 0:
        df_clean.loc['tikka masala', 'label'] = 1
        changes += 1
    
    # Add missing common dishes
    missing_dishes = ['naan', 'dal', 'samosa', 'biryani', 'curry', 'paneer', 'lassi', 'masala']
    
    for dish in missing_dishes:
        if dish not in df_clean.index:
            df_clean.loc[dish] = 1
            changes += 1
    
    print(f"Made {changes} changes")
    print(f"Before: {sum(df['label'] == 1)} dishes")
    print(f"After: {sum(df_clean['label'] == 1)} dishes")
    
    return df_clean

def get_indian_reviews(business_file, review_file):
    print("Loading Indian restaurant reviews...")
    
    businesses = pd.read_json(business_file, lines=True, dtype={"categories": "string"})
    businesses['categories'] = businesses['categories'].fillna("")
    
    is_restaurant = businesses['categories'].str.contains('Restaurants', case=False, na=False)
    is_indian = businesses['categories'].str.contains('Indian', case=False, na=False)
    
    indian_business_ids = set(businesses.loc[is_restaurant & is_indian]['business_id'])
    print(f"Found {len(indian_business_ids)} Indian restaurants")
    
    reviews = []
    count = 0
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= 5000:
                break
            review = json.loads(line)
            if review['business_id'] in indian_business_ids:
                reviews.append(review)
                count += 1
    
    print(f"Collected {len(reviews)} reviews")
    return pd.DataFrame(reviews)

def mine_dish_patterns(reviews, known_dishes):
    print("Mining for new dishes...")
    
    if reviews.empty:
        print("No reviews to analyze")
        return []
    
    all_text = ' '.join(reviews['text'].astype(str)).lower()
    
    # Pattern matching for dish mentions
    patterns = [
        r'ordered (?:the )?([a-z][a-z\s]{2,20}?)(?:\s+(?:and|with|was|is|\.|\,))',
        r'tried (?:the )?([a-z][a-z\s]{2,20}?)(?:\s+(?:and|with|was|is|\.|\,))',
        r'had (?:the )?([a-z][a-z\s]{2,20}?)(?:\s+(?:and|with|was|is|\.|\,))',
        r'delicious ([a-z][a-z\s]{2,20}?)(?:\s+(?:and|with|was|is|\.|\,))',
        r'spicy ([a-z][a-z\s]{2,20}?)(?:\s+(?:and|with|was|is|\.|\,))',
    ]
    
    candidates = Counter()
    
    for pattern in patterns:
        matches = re.findall(pattern, all_text)
        for match in matches:
            match = match.strip()
            if (2 <= len(match.split()) <= 3 and 
                match not in known_dishes and
                'service' not in match and 'place' not in match):
                candidates[match] += 1
    
    # Filter by frequency
    new_dishes = [(dish, count) for dish, count in candidates.items() if count >= 3]
    new_dishes.sort(key=lambda x: x[1], reverse=True)
    new_dishes = new_dishes[:20]
    
    print(f"Found {len(new_dishes)} new candidates")
    return new_dishes

def create_visualizations(original, cleaned, new_candidates, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before/after comparison
    before_counts = [sum(original['label'] == 1), sum(original['label'] == 0)]
    after_counts = [sum(cleaned['label'] == 1), sum(cleaned['label'] == 0)]
    
    categories = ['Dishes', 'Not Dishes']
    x_pos = [0, 1]
    width = 0.35
    
    ax1.bar([p - width/2 for p in x_pos], before_counts, width, label='Before', alpha=0.7)
    ax1.bar([p + width/2 for p in x_pos], after_counts, width, label='After', alpha=0.7)
    ax1.set_ylabel('Count')
    ax1.set_title('Label Cleaning Results')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories)
    ax1.legend()
    
    # New dishes discovered
    if new_candidates:
        dishes, counts = zip(*new_candidates[:10])
        ax2.barh(range(len(dishes)), counts, color='orange')
        ax2.set_yticks(range(len(dishes)))
        ax2.set_yticklabels(dishes)
        ax2.set_xlabel('Mentions')
        ax2.set_title('New Dishes Discovered')
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'No new dishes found', ha='center', va='center')
        ax2.set_title('New Dishes Discovered')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return plot_path

def main():
    base_dir = r'C:\Users\dheer\OneDrive\Desktop\Coursera Data Mining\yelp_dataset_challenge_academic_dataset'
    labels_dir = r'C:\Users\dheer\OneDrive\Desktop\Coursera Data Mining\manualAnnotationTask'
    business_file = os.path.join(base_dir, 'yelp_academic_dataset_business.json')
    review_file = os.path.join(base_dir, 'yelp_academic_dataset_review.json')
    
    save_dir = r'C:\Users\dheer\OneDrive\Desktop\Coursera Data Mining\Visuals\task3'
    os.makedirs(save_dir, exist_ok=True)
    
    indian_file = os.path.join(labels_dir, 'Indian.label')
    
    print("Task 3: Finding Indian dishes")
    print("=" * 30)
    
    # Load original labels
    original_labels = load_labels(indian_file)
    print(f"Loaded {len(original_labels)} original labels")
    print(f"Originally marked as dishes: {sum(original_labels['label'] == 1)}")
    
    # Task 3.1: Clean labels
    print("\nTask 3.1: Manual label cleaning")
    cleaned_labels = clean_dish_labels(original_labels)
    
    verified_dishes = cleaned_labels[cleaned_labels['label'] == 1].index.tolist()
    print(f"Verified dishes: {verified_dishes}")
    
    # Task 3.2: Mine for additional dishes
    print("\nTask 3.2: Mining reviews for new dishes")
    review_data = get_indian_reviews(business_file, review_file)
    discovered_dishes = mine_dish_patterns(review_data, verified_dishes)
    
    # Create visualizations
    plot_path = create_visualizations(original_labels, cleaned_labels, discovered_dishes, save_dir)
    
    # Save cleaned labels
    output_file = os.path.join(save_dir, 'Indian_cleaned.label')
    with open(output_file, 'w') as f:
        for dish, row in cleaned_labels.iterrows():
            f.write(f"{dish}\t{row['label']}\n")
    
    # Save summary
    summary_file = os.path.join(save_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Task 3 Results\n")
        f.write("=" * 15 + "\n\n")
        
        f.write("Task 3.1 - Manual Cleaning:\n")
        f.write(f"Started with: {len(original_labels)} labels\n")
        f.write(f"Ended with: {len(cleaned_labels)} labels\n")
        f.write(f"Verified dishes: {sum(cleaned_labels['label'] == 1)}\n\n")
        
        f.write("Cleaned dish list:\n")
        for dish in sorted(verified_dishes):
            f.write(f"  {dish}\n")
        
        f.write(f"\nTask 3.2 - Pattern Mining:\n")
        f.write(f"Discovered {len(discovered_dishes)} new candidates\n\n")
        
        if discovered_dishes:
            f.write("New candidates:\n")
            for dish, count in discovered_dishes:
                f.write(f"  {dish} ({count} mentions)\n")
    
    print(f"\nResults saved:")
    print(f"  Cleaned labels: {output_file}")
    print(f"  Summary: {summary_file}")
    print(f"  Visualization: {plot_path}")

if __name__ == "__main__":
    main()