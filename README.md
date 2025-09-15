# Yelp Data Mining Project

Data mining and analysis of Yelp reviews to discover cuisines, popular dishes, restaurant recommendations, and hygiene predictions.

## Overview
This project applies data mining techniques to the Yelp dataset to extract insights that support better dining decisions.  
The work includes topic modeling, cuisine mapping, dish recognition, popularity analysis, restaurant recommendations, and hygiene prediction.  
Deliverables include Python scripts, reports, and visual outputs.

## Repository Structure
- `scripts/` – Python scripts for each task:
  - `task1.py`
  - `task2.py`
  - `task3.py`
  - `task4_5.py`
  - `task6.py`
- `reports/` – Individual task reports and the final report (PDFs)
- `report_overview/` – Overview documents (PDFs)
- `visuals/` – Figures and charts
- `data/` – Placeholder for Yelp dataset (not included in repository)

## Project Tasks
### Task 1 – Exploratory Topic Modeling
- Extracted topics from Yelp reviews using LDA/PLSA.  
- Compared topics across positive and negative reviews.  
- Visualized distributions of ratings and review patterns.  
**Files**: [Report](reports/Task%201.pdf) | [Code](scripts/task1.py)

### Task 2 – Cuisine Map Construction
- Represented cuisines using aggregated restaurant reviews.  
- Computed cuisine similarities (TF-IDF, cosine similarity, LDA).  
- Generated a cuisine similarity map with clustering.  
**Files**: [Report](reports/Task%202.pdf) | [Code](scripts/task2.py)

### Task 3 – Dish Recognition
- Built and refined a dish name recognizer using SegPhrase, ToPMine, and word2vec.  
- Identified candidate dishes and expanded with automatic labeling.  
**Files**: [Report](reports/Task%203.pdf) | [Code](scripts/task3.py)

### Task 4 – Mining Popular Dishes
- Ranked dishes by frequency and sentiment in reviews.  
- Produced visualizations of popular dishes by cuisine.  
**Files**: [Report](reports/Task%204.pdf) | [Code](scripts/task4_5.py)

### Task 5 – Restaurant Recommendation
- Recommended restaurants based on dish-specific mentions.  
- Designed ranking functions combining frequency and sentiment.  
**Files**: [Report](reports/Task%205.pdf) | [Code](scripts/task4_5.py)

### Task 6 – Hygiene Prediction
- Predicted restaurant health inspection outcomes.  
- Used textual and non-textual features (cuisine, location, ratings).  
- Evaluated models with F1-score to address class imbalance.  
**Files**: [Report](reports/Task%206.pdf) | [Code](scripts/task6.py)

### Final Report
- Summarized all findings.  
- Highlighted contributions and practical implications.  
**Files**: [Final Report](reports/Final%20Report.pdf)

## Technical Stack
- **Languages/Tools**: Python, Pandas, NumPy, Matplotlib, Scikit-learn, NLTK, TextBlob  
- **Techniques**: Topic modeling (LDA/PLSA), TF-IDF, clustering, word embeddings, phrase mining  
- **Models**: Logistic Regression, SVM, Naïve Bayes, ensemble classifiers  

## Usage
Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/yelp-data-mining-project.git
cd yelp-data-mining-project
pip install -r requirements.txt
