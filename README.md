NLP: Word Representations and Text Classification
This project explores the transition from traditional probabilistic text models to dense vector representations. It features a from-scratch implementation of the Skip-Gram Word2Vec model and a comparative study of Generative and Discriminative classification architectures.

🚀 Key Features
1. Custom Word2Vec Implementation (Skip-Gram)
Designed and trained a neural word embedding model to capture semantic relationships.

Negative Sampling: Implemented Negative Sampling to bypass the computational bottleneck of the full Softmax denominator, reframing the problem as a binary classification task.

Forward Pass: Developed the manual forward pass logic in PyTorch, calculating dot-product similarity scores between center, positive context, and negative noise words.

Training Pipeline: Built a robust trainer with a linear learning rate scheduler and gradient clipping to ensure stable convergence.

2. Generative Modeling (Naive Bayes)Implemented a probabilistic classifier based on the "Naive" conditional independence assumption.Preprocessing: Developed a pipeline for tokenization, stop-word removal, and frequency-based vocabulary pruning.Laplace Smoothing: Incorporated alpha-smoothing (P(X|y) = {count(X,y) + alpha}/{total_words + alpha|V|} to handle the zero-probability problem for unseen words.Feature Importance: Created a ranking system using probability ratios (P(word | c) / avg(P(word | others))) to identify the most distinctive words for each class.
3. Discriminative Classification
Compared high-dimensional sparse representations with low-dimensional dense embeddings.

Bag-of-Words (BoW): Built a multiclass logistic regression model using document-term matrices.

Word2Vec-Softmax: Leveraged pre-trained Google News 300 embeddings and document mean-pooling to achieve semantic-aware classification.

📊 Methodology & Evaluation
Similarity Functions
To evaluate the quality of the learned embeddings, the project implements two primary distance metrics:

Cosine Similarity: Measures the angular relationship between vectors, focusing on orientation rather than magnitude.

Euclidean Distance: Measures the straight-line distance between points in the vector space.

Dataset
The models are trained and validated on a news dataset consisting of five categories: Politics, Entertainment, Sports, Business, and Tech.

🛠️ Tech Stack
Language: Python

Deep Learning: PyTorch

Data Science: NumPy, Pandas, Scikit-learn

NLP Tools: Gensim (for pre-trained Word2Vec), NLTK (for stop-words)

📂 Repository Structure
Satyajeet_Kumar_26813_assignment1.py: Main implementation file.

weights: Directory containing saved model states (.pt and .pkl).

README.md: Project documentation.

Developed as part of the DS 207: Intro to NLP course at the Indian Institute of Science (IISc), Bengaluru.
