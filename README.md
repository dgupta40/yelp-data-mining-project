# Yelp Data Mining Project  

Data mining and analysis of Yelp reviews to discover cuisines, popular dishes, restaurant recommendations, and hygiene predictions.  

---

## 📌 Overview  
This project applies end-to-end **data mining techniques** on the Yelp dataset (1.6M reviews, 61k businesses, 366k users).  
The goal is to extract actionable insights to help people make better dining decisions.  

We implemented tasks ranging from **topic modeling** to **predictive modeling**, producing both **visualizations** and **reports**.  

---

## 🗂️ Project Tasks  

### **Task 1 – Exploratory Topic Modeling**  
- Extracted topics from Yelp reviews using **LDA/PLSA**.  
- Compared topics across **positive vs negative reviews**.  
- Visualized distributions of ratings and review patterns.  
📄 [Report](reports/Task1.pdf) | 💻 [Code](code/task1.py)  

---

### **Task 2 – Cuisine Map Construction**  
- Represented cuisines using aggregated restaurant reviews.  
- Computed cuisine similarities (TF-IDF, LDA, cosine similarity).  
- Visualized a **cuisine similarity map** with clustering.  
📄 [Report](reports/Task2.pdf) | 💻 [Code](code/task2.py)  

---

### **Task 3 – Dish Recognition**  
- Refined candidate dish names using manual + automatic labeling.  
- Expanded dish list using **SegPhrase, ToPMine, word2vec**.  
- Built a **dish recognizer** to identify popular items per cuisine.  
📄 [Report](reports/Task3.pdf) | 💻 [Code](code/task3.py)  

---

### **Task 4 – Mining Popular Dishes**  
- Ranked dishes by **frequency + sentiment of mentions** in reviews.  
- Generated visualizations of **top dishes per cuisine**.  
📄 [Report](reports/Task4_5.pdf) | 💻 [Code](code/task4_5.py)  

---

### **Task 5 – Restaurant Recommendation**  
- Recommended restaurants based on **dish-specific reviews**.  
- Designed ranking functions combining mentions + sentiment.  
- Produced **dish-aware restaurant rankings**.  
📄 [Report](reports/Task4_5.pdf) | 💻 [Code](code/task4_5.py)  

---

### **Task 6 – Hygiene Prediction**  
- Predicted whether restaurants pass **public health inspections**.  
- Combined **textual features** (reviews) with **non-textual data** (location, cuisine, ratings).  
- Evaluated classifiers using **F1-score** to handle class imbalance.  
📄 [Report](reports/Task6.pdf) | 💻 [Code](code/task6.py)  

---

### **Final Report**  
- Summarized all findings.  
- Highlighted **usefulness of results**, **novel contributions**, and **new insights**.  
📄 [Final Report](reports/Final_Report.pdf)  

---

## 📊 Key Results  
- Cuisine maps revealed **clusters of related cuisines** (e.g., Indian ↔ Pakistani).  
- Dish recognizer surfaced both **expected favorites** and **hidden gems**.  
- Restaurant recommender enabled **dish-specific choices** instead of generic ratings.  
- Hygiene prediction models achieved strong performance on imbalanced data.  

---

## 🛠️ Tech Stack  
- **Languages/Tools**: Python, Jupyter, Pandas, NumPy, Matplotlib, Seaborn  
- **Algorithms**: LDA, PLSA, TF-IDF, clustering (k-means, hierarchical), word2vec, SegPhrase, ToPMine  
- **ML Models**: Logistic Regression, SVM, Naive Bayes, ensemble classifiers  

---

## ▶️ Usage  
Clone the repository and install dependencies:  

```bash
git clone https://github.com/<your-username>/yelp-data-mining-project.git
cd yelp-data-mining-project
pip install -r requirements.txt
