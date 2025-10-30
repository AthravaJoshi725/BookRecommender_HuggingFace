---
title: Semantic Book Recommender
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: AI book recommender using semantic search and emotions
---

# 📚 Semantic Book Recommender

A machine learning-powered book recommendation system that provides personalized book suggestions based on **content similarity and sentiment analysis**.

## 🔍 How It Works

1. **Semantic Search**: Uses FAISS vector search with HuggingFace embeddings to find books with similar content
2. **Emotion Analysis**: Analyzes emotional tones in book descriptions (Happy, Surprising, Angry, Suspenseful, Sad)
3. **Category Filtering**: Filter recommendations by book categories
4. **Smart Recommendations**: Combines semantic similarity with emotional preferences

## 🚀 Features

- **📖 Content-Based Recommendations**: Find books similar to your description
- **🎭 Emotion-Based Filtering**: Get books that match your desired emotional tone
- **📂 Category Filtering**: Filter by book categories
- **🖼️ Visual Interface**: Browse recommendations with book covers and descriptions

## 🛠️ Technology Stack

- **Gradio** - Interactive web interface
- **FAISS** - Vector similarity search
- **HuggingFace Transformers** - Text embeddings and emotion analysis
- **LangChain** - Document processing and vector operations

## 💡 Usage

1. **Describe a book** you're looking for (e.g., "A story about forgiveness")
2. **Select a category** (optional) - Fiction, Non-Fiction, etc.
3. **Choose emotional tone** (optional) - Happy, Suspenseful, Sad, etc.
4. **Get recommendations** - Browse through AI-curated book suggestions

## 🔧 Technical Details

- **Embedding Model**: `sentence-transformers/paraphrase-MiniLM-L6-v2`
- **Vector Store**: FAISS for efficient similarity search
- **Dataset**: Books with descriptions, categories, and emotion scores
- **Pre-built Index**: Optimized for fast startup times

---

*Built with ❤️ using Gradio and HuggingFace Transformers*