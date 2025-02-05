# 📚 Book Recommender System

A machine-learning-powered book recommendation system that provides personalized book suggestions based on **content similarity and sentiment analysis**.

---

## 🔍 Project Overview

This project focuses on **analyzing book descriptions**, extracting meaningful features, and performing **vector-based similarity search** to recommend books. It also classifies books into **Fiction** or **Non-Fiction** and analyzes emotions in book descriptions.

---

## 🚀 Key Features

- **📖 Data Cleaning & EDA:** Preprocessed dataset by handling missing values, creating new features, and analyzing book descriptions.
- **🧠 Vector Search with FAISS:** Used **FAISS** to store book embeddings and perform similarity-based recommendations.
- **🎭 Text Classification (Zero-Shot Learning):** Classified books as **Fiction/Non-Fiction** using **Facebook BART model**.
- **😊 Sentiment & Emotion Analysis:** Detected emotional content in book descriptions using **J-Hartmann’s emotion classification model**.
- **💻 Interactive UI:** Integrated **Gradio** for an easy-to-use interface for recommendations.

---

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Libraries Used:**
  - `Pandas`, `NumPy` → Data manipulation
  - `Matplotlib`, `Seaborn` → Data visualization
  - `FAISS` → Vector similarity search
  - `Hugging Face Transformers` → Embeddings & classification
  - `Gradio` → Interactive web UI

---

## 🏗️ Project Workflow

1. **📊 Data Preprocessing & EDA:**

   - Loaded dataset from Kaggle, cleaned missing values, and engineered new features like `Age of Book`.
   - Analyzed book descriptions to remove **unhelpful** ones.

2. **🔎 Vector Search for Recommendations:**

   - Converted descriptions into **embeddings** using **Hugging Face models**.
   - Stored embeddings in **FAISS** for efficient similarity search.
   - Used ISBN as an identifier to find books with **similar content**.

3. **🏷️ Text Classification (Fiction vs. Non-Fiction):**

   - The dataset had **500+ categories**, so we simplified it to just **Fiction & Non-Fiction**.
   - Used **Zero-Shot Classification** via the **Facebook BART model** to categorize books.

4. **💬 Sentiment & Emotion Analysis:**

   - Used **J-Hartmann’s emotion classifier** to detect **emotional tones** in descriptions.
   - **Problem:** Single-label classification missed important emotions.
   - **Solution:** Split descriptions into sentences and analyzed each separately.
   - **Final Output:** Each book now has **multiple emotion scores** (e.g., Anger, Joy, Sadness).

5. **🌐 User Interface with Gradio:**
   - Built a **simple UI** using **Gradio** to allow users to input book descriptions and get recommendations.

---

## 🎯 Why These Models?

- **Hugging Face Embeddings** → Converts text descriptions into **meaningful vectors** for similarity search.
- **FAISS (Facebook AI Similarity Search)** → Efficiently finds **similar books** based on embeddings.
- **Facebook BART (Zero-Shot Classification)** → Classifies books **without needing pre-labeled training data**.
- **J-Hartmann Emotion Classifier** → Extracts **multiple emotions** from text for better sentiment analysis.

---

## 🖥️ How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/book-recommender.git
   cd book-recommender
   ```
