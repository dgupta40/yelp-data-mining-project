#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re
import json
import os
from collections import Counter
import matplotlib.pyplot as plt
from textblob import TextBlob

def load_dish_list():
    # Verified dishes from manual cleaning (Task 3.1)
    verified_dishes = [
        'basmati rice', 'biryani', 'brown rice', 'chick peas', 'chicken tikka',
        'chicken tikka masala', 'chicken wings', 'curry', 'dal', 'flat bread',
        'fried rice', 'gluten free', 'gulab jamun', 'lassi', 'masala', 'naan',
        'paneer', 'rice pudding', 'rogan josh', 'samosa', 'tandoori chicken',
        'tikka masala', 'tomato sauce', 'tomato soup', 'white rice'
    ]
    
    # Newly discovered dishes from pattern mining (Task 3.2)
    discovered_dishes = [
        'lamb vindaloo', 'chicken curry', 'garlic naan', 'lamb korma', 
        'palak paneer', 'chicken korma', 'chicken vindaloo', 'butter chicken',
        'lamb curry', 'chicken masala', 'lamb biryani', 'chicken biryani'
    ]
    
    all_dishes = verified_dishes + discovered_dishes
    print(f"Analyzing {len(all_dishes)} dishes total")
    return all_dishes

def load_review_data(business_file, review_file):
    print("Loading data...")
    
    businesses = pd.read_json(business_file, lines=True, dtype={"categories": "string"})
    businesses['categories'] = businesses['categories'].fillna("")
    
    is_restaurant = businesses['categories'].str.contains('Restaurants', case=False, na=False)
    is_indian = businesses['categories'].str.contains('Indian', case=False, na=False)
    
    indian_business_ids = set(businesses.loc[is_restaurant & is_indian]['business_id'])
    print(f"Found {len(indian_business_ids)} Indian restaurants")
    
    # Store restaurant metadata
    restaurant_info = {}
    for _, row in businesses.loc[is_restaurant & is_indian].iterrows():
        restaurant_info[row['business_id']] = {
            'name': row['name'],
            'stars': row['stars']
        }
    
    # Load reviews
    reviews = []
    count = 0
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= 8000:
                break
            review = json.loads(line)
            if review['business_id'] in indian_business_ids:
                reviews.append({
                    'business_id': review['business_id'],
                    'stars': review['stars'],
                    'text': review['text']
                })
                count += 1
    
    print(f"Loaded {len(reviews)} reviews")
    return reviews, restaurant_info

def extract_dish_mentions(reviews, dishes):
    print("Extracting dish mentions...")
    
    mentions = []
    
    for review in reviews:
        text = review['text'].lower()
        
        for dish in dishes:
            if dish.lower() in text:
                sentiment = 0
                try:
                    sentiment = TextBlob(text).sentiment.polarity
                except:
                    pass
                
                mentions.append({
                    'dish': dish,
                    'business_id': review['business_id'],
                    'review_stars': review['stars'],
                    'sentiment': sentiment
                })
    
    print(f"Found {len(mentions)} dish mentions")
    return mentions

def calculate_dish_popularity(mentions):
    print("Calculating dish popularity...")
    
    dish_stats = {}
    
    for dish in set([m['dish'] for m in mentions]):
        dish_mentions = [m for m in mentions if m['dish'] == dish]
        
        if len(dish_mentions) == 0:
            continue
            
        total_mentions = len(dish_mentions)
        avg_sentiment = sum([m['sentiment'] for m in dish_mentions]) / len(dish_mentions)
        avg_stars = sum([m['review_stars'] for m in dish_mentions]) / len(dish_mentions)
        num_restaurants = len(set([m['business_id'] for m in dish_mentions]))
        
        # Composite scoring function
        score = (total_mentions * 0.4 + 
                (avg_sentiment + 1) * 10 * 0.3 + 
                avg_stars * 0.2 + 
                num_restaurants * 0.1)
        
        dish_stats[dish] = {
            'total_mentions': total_mentions,
            'avg_sentiment': avg_sentiment,
            'avg_stars': avg_stars,
            'num_restaurants': num_restaurants,
            'score': score
        }
    
    ranked_dishes = sorted(dish_stats.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print("Top dishes:")
    for i, (dish, data) in enumerate(ranked_dishes[:10]):
        print(f"  {i+1}. {dish} - {data['score']:.1f} ({data['total_mentions']} mentions)")
    
    return ranked_dishes

def rank_restaurants_for_dish(mentions, restaurant_info, dish_name):
    print(f"Ranking restaurants for: {dish_name}")
    
    dish_mentions = [m for m in mentions if m['dish'] == dish_name]
    
    if not dish_mentions:
        print("No mentions found")
        return []
    
    restaurant_scores = {}
    
    for business_id in set([m['business_id'] for m in dish_mentions]):
        restaurant_mentions = [m for m in dish_mentions if m['business_id'] == business_id]
        
        mention_count = len(restaurant_mentions)
        avg_sentiment = sum([m['sentiment'] for m in restaurant_mentions]) / len(restaurant_mentions)
        avg_stars = sum([m['review_stars'] for m in restaurant_mentions]) / len(restaurant_mentions)
        
        score = (mention_count * 0.3 + 
                (avg_sentiment + 1) * 10 * 0.4 + 
                avg_stars * 0.3)
        
        restaurant_scores[business_id] = {
            'name': restaurant_info.get(business_id, {}).get('name', 'Unknown'),
            'mention_count': mention_count,
            'avg_sentiment': avg_sentiment,
            'avg_stars': avg_stars,
            'score': score
        }
    
    ranked_restaurants = sorted(restaurant_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print(f"Top restaurants:")
    for i, (_, data) in enumerate(ranked_restaurants[:5]):
        print(f"  {i+1}. {data['name']} - {data['score']:.1f}")
    
    return ranked_restaurants

def visualize_dish_rankings(ranked_dishes, save_dir):
    top_dishes = ranked_dishes[:15]
    dish_names = [d[0] for d in top_dishes]
    scores = [d[1]['score'] for d in top_dishes]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(dish_names)), scores)
    plt.yticks(range(len(dish_names)), dish_names)
    plt.xlabel('Popularity Score')
    plt.title('Popular Indian Dishes Ranked by Composite Score')
    plt.gca().invert_yaxis()
    
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(scores[i] / max(scores)))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'task4_dishes.png'), dpi=300, bbox_inches='tight')
    plt.show()

def visualize_restaurant_rankings(ranked_restaurants, dish_name, save_dir):
    top_restaurants = ranked_restaurants[:10]
    restaurant_names = [r[1]['name'][:20] for r in top_restaurants]
    scores = [r[1]['score'] for r in top_restaurants]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(restaurant_names)), scores)
    plt.yticks(range(len(restaurant_names)), restaurant_names)
    plt.xlabel('Recommendation Score')
    plt.title(f'Best Restaurants for {dish_name}')
    plt.gca().invert_yaxis()
    
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.plasma(scores[i] / max(scores)))
    
    plt.tight_layout()
    filename = dish_name.replace(' ', '_') + '.png'
    plt.savefig(os.path.join(save_dir, f'task5_{filename}'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    base_dir = r'C:\Users\dheer\OneDrive\Desktop\Coursera Data Mining\yelp_dataset_challenge_academic_dataset'
    save_dir = r'C:\Users\dheer\OneDrive\Desktop\Coursera Data Mining\Visuals\tasks_4_5'
    
    os.makedirs(save_dir, exist_ok=True)
    
    business_file = os.path.join(base_dir, 'yelp_academic_dataset_business.json')
    review_file = os.path.join(base_dir, 'yelp_academic_dataset_review.json')
    
    print("Starting Tasks 4 and 5...")
    
    # Load dish list and review data
    dishes = load_dish_list()
    reviews, restaurant_info = load_review_data(business_file, review_file)
    mentions = extract_dish_mentions(reviews, dishes)
    
    # Task 4: Rank popular dishes
    print("\nTask 4: Ranking popular dishes")
    dish_rankings = calculate_dish_popularity(mentions)
    visualize_dish_rankings(dish_rankings, save_dir)
    
    # Task 5: Restaurant recommendations for top dishes
    print("\nTask 5: Restaurant recommendations")
    top_dishes = [dish_rankings[i][0] for i in range(min(3, len(dish_rankings)))]
    
    for dish in top_dishes:
        restaurant_rankings = rank_restaurants_for_dish(mentions, restaurant_info, dish)
        if restaurant_rankings:
            visualize_restaurant_rankings(restaurant_rankings, dish, save_dir)
    
    # Save results summary
    output_file = os.path.join(save_dir, 'output.txt')
    with open(output_file, 'w') as f:
        f.write("Tasks 4 & 5 Results\n")
        f.write("=" * 20 + "\n\n")
        
        f.write("Task 4 - Popular Dishes:\n")
        for i, (dish, data) in enumerate(dish_rankings[:15]):
            f.write(f"{i+1}. {dish} - Score: {data['score']:.1f}, Mentions: {data['total_mentions']}\n")
        
        f.write(f"\nTask 5 - Restaurant Recommendations:\n")
        for dish in top_dishes:
            restaurant_rankings = rank_restaurants_for_dish(mentions, restaurant_info, dish)
            f.write(f"\nBest restaurants for {dish}:\n")
            for i, (_, data) in enumerate(restaurant_rankings[:5]):
                f.write(f"  {i+1}. {data['name']} - Score: {data['score']:.1f}\n")
    
    print(f"\nAnalysis complete! Results saved to {save_dir}")

if __name__ == "__main__":
    main()