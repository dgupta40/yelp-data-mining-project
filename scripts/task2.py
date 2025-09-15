#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def parse_categories(cat_str):
    if pd.isna(cat_str) or cat_str == "":
        return []
    
    cat_str = cat_str.strip("[]'\"")
    parts = [c.strip().strip("'\"[]") for c in cat_str.split(',')]
    return [c for c in parts if c and c.lower() != 'restaurants']

def plot_heatmap(similarity_matrix, cuisine_labels, title, cluster_colors=None, save_path=None):
    n = similarity_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(cuisine_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(cuisine_labels, fontsize=6)
    
    if cluster_colors is not None:
        for tick_label, color in zip(ax.get_xticklabels(), cluster_colors):
            tick_label.set_color(color)
        for tick_label, color in zip(ax.get_yticklabels(), cluster_colors):
            tick_label.set_color(color)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def assign_cluster_colors(cluster_labels, colormap_name='tab10'):
    unique_labels = np.unique(cluster_labels)
    cmap = plt.colormaps.get_cmap(colormap_name).resampled(len(unique_labels))
    color_map = {lab: cmap(i) for i, lab in enumerate(unique_labels)}
    
    def rgba_to_hex(rgba):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
        )
    return [rgba_to_hex(color_map[lab]) for lab in cluster_labels]

def is_valid_cuisine(cuisine_name):
    if pd.isna(cuisine_name) or cuisine_name == "":
        return False
    
    if any(char in cuisine_name for char in ["']", "['", "'", '"']):
        return False
        
    excluded = ['nightlife', 'bars', 'lounges', 'pubs', 'beer', 'wine', 
               'cocktail', 'sports bars', 'dive bars', 'gay bars']
    if cuisine_name.lower() in excluded:
        return False
        
    return True

def main():
    base_dir = r'C:\Users\dheer\OneDrive\Desktop\Coursera Data Mining\yelp_dataset_challenge_academic_dataset'
    business_file = os.path.join(base_dir, 'yelp_academic_dataset_business.json')
    review_file = os.path.join(base_dir, 'yelp_academic_dataset_review.json')
    
    save_dir = r'C:\Users\dheer\OneDrive\Desktop\Coursera Data Mining\Visuals'
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading business data...")
    businesses = pd.read_json(business_file, lines=True, dtype={"categories": "string"})
    
    businesses['categories'] = businesses['categories'].fillna("")
    restaurant_mask = businesses['categories'].str.contains('Restaurants', case=False, na=False)
    businesses = businesses.loc[restaurant_mask].copy()
    
    businesses['cuisine_list'] = businesses['categories'].apply(parse_categories)
    
    print("Loading reviews...")
    reviews = pd.read_json(review_file, lines=True, dtype={"business_id": "string", "text": "string"})
    
    merged_data = reviews.merge(
        businesses[['business_id', 'cuisine_list']],
        on='business_id',
        how='inner'
    )
    
    exploded_data = merged_data.explode('cuisine_list').rename(columns={'cuisine_list': 'cuisine'})
    
    cuisine_texts = (
        exploded_data
        .groupby('cuisine', as_index=False)['text']
        .agg(combined_reviews=lambda texts: " ".join(texts))
    )
    
    # Filter cuisines
    cuisine_texts['char_count'] = cuisine_texts['combined_reviews'].str.len()
    valid_cuisines = cuisine_texts[cuisine_texts['cuisine'].apply(is_valid_cuisine)].copy()
    
    MIN_CHARS = 500_000
    large_enough = valid_cuisines[valid_cuisines['char_count'] >= MIN_CHARS]
    
    MAX_CUISINES = 60
    if len(large_enough) > MAX_CUISINES:
        final_cuisines = large_enough.nlargest(MAX_CUISINES, 'char_count')
    else:
        final_cuisines = large_enough
    
    print(f"Analyzing {len(final_cuisines)} cuisines")
    
    # Create feature matrices
    vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=2000)
    count_matrix = vectorizer.fit_transform(final_cuisines["combined_reviews"])
    
    tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True)
    tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)
    
    cuisine_names = final_cuisines["cuisine"].tolist()
    
    # Raw TF similarity
    count_normalized = normalize(count_matrix, norm='l2', axis=1)
    tf_similarity = count_normalized.dot(count_normalized.T).toarray()
    
    plot_heatmap(
        tf_similarity,
        cuisine_names,
        title="Cuisine Similarity (Raw TF) - Cosine Heatmap",
        save_path=os.path.join(save_dir, "cuisine_similarity_raw_tf.png")
    )
    
    # TF-IDF similarity
    tfidf_normalized = normalize(tfidf_matrix, norm='l2', axis=1)
    tfidf_similarity = tfidf_normalized.dot(tfidf_normalized.T).toarray()
    
    plot_heatmap(
        tfidf_similarity,
        cuisine_names,
        title="Cuisine Similarity (TF-IDF) - Cosine Heatmap",
        save_path=os.path.join(save_dir, "cuisine_similarity_tfidf.png")
    )
    
    # LDA topic modeling
    n_topics = 10
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        learning_method='online',
        random_state=42,
        batch_size=128
    )
    lda_model.fit(count_matrix)
    
    topic_distributions = lda_model.transform(count_matrix)
    topic_distributions = normalize(topic_distributions, norm='l1', axis=1)
    
    lda_similarity = np.dot(topic_distributions, topic_distributions.T)
    
    plot_heatmap(
        lda_similarity,
        cuisine_names,
        title=f"Cuisine Similarity (LDA, {n_topics} topics)",
        save_path=os.path.join(save_dir, "cuisine_similarity_lda.png")
    )
    
    # Display topics
    feature_names = vectorizer.get_feature_names_out()
    print(f"\nTopic keywords:")
    for i, topic_weights in enumerate(lda_model.components_):
        top_indices = topic_weights.argsort()[::-1][:10]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"Topic {i}: {', '.join(top_words)}")
    
    # Clustering analysis
    cluster_results = {}
    
    for algorithm, ClusterClass in [('KMeans', KMeans), ('Agglomerative', AgglomerativeClustering)]:
        for k in [4, 8]:
            if algorithm == 'KMeans':
                clusterer = ClusterClass(n_clusters=k, random_state=42)
                labels = clusterer.fit_predict(topic_distributions)
            else:
                clusterer = ClusterClass(n_clusters=k, metric='cosine', linkage='average')
                labels = clusterer.fit_predict(topic_distributions)
            
            cluster_results[f"{algorithm}_k{k}"] = labels
            
            colors = assign_cluster_colors(labels, colormap_name='tab10')
            
            filename = f"cuisine_similarity_lda_{algorithm.lower()}_k{k}.png"
            plot_heatmap(
                lda_similarity,
                cuisine_names,
                title=f"{algorithm} (k={k}) on LDA-space: Cosine Heatmap",
                cluster_colors=colors,
                save_path=os.path.join(save_dir, filename)
            )
    
    # Evaluation metrics
    print("\nClustering evaluation:")
    for key, labels in cluster_results.items():
        score = silhouette_score(topic_distributions, labels)
        print(f"{key}: Silhouette Score = {score:.3f}")
    
    # Validation check
    print("\nValidation test:")
    indian_idx = None
    for i, cuisine in enumerate(cuisine_names):
        if 'indian' in cuisine.lower():
            indian_idx = i
            print(f"Found Indian cuisine at index {i}: '{cuisine}'")
            print(f"Topic distribution: {topic_distributions[i]}")
            break
    
    if indian_idx is None:
        print("Indian cuisine not found in dataset")
    
    # Save results
    results_df = final_cuisines.copy().reset_index(drop=True)
    
    for topic_idx in range(n_topics):
        results_df[f"topic_{topic_idx}"] = topic_distributions[:, topic_idx]
    
    for key, labels in cluster_results.items():
        results_df[key] = labels
    
    output_file = os.path.join(base_dir, f'task2_results_{len(final_cuisines)}_cuisines.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"Visualizations saved to {save_dir}")
    print(f"Data saved to {output_file}")
    print("Task 2 complete!")

if __name__ == "__main__":
    main()