# Financial Sentiment Analysis 

This project is developed for the **Methods of Artificial Intelligence** course. It aims to build an NLP (Natural Language Processing) classification model that can read financial news headlines and categorize their sentiment as **Positive**, **Negative**, or **Neutral**.

To demonstrate a rigorous understanding of AI methods, this project compares two distinct approaches:
1. **Traditional Machine Learning (Baseline):** Using TF-IDF vectorization and a simple linear classifier via `scikit-learn`.
2. **Deep Sequence Learning:** Using an Embedding layer and an LSTM (Long Short-Term Memory) neural network built with `tensorflow` and `keras`.

## The Dataset
The project trains on the "Financial PhraseBank" dataset, which contains sentences from English financial news categorized by sentiment. 

**Data Source:** [Sentiment Analysis for Financial News (Kaggle)](https://www.kaggle.com/code/prasadchaskar/sentiment-analysis-for-financial-news)

*Note: The raw CSV file is not uploaded to this repository to keep it lightweight. To run this code locally, download the dataset from Kaggle, rename it to `all-data.csv`, and place it in a folder named `Data/` in the root directory.*

## Technologies & Libraries Used
* **Python 3.11**
* **Pandas & NumPy** (Data manipulation)
* **Scikit-Learn** (Label encoding, TF-IDF, Baseline models)
* **TensorFlow / Keras** (Deep learning architecture)
* **Matplotlib & Seaborn** (Performance visualization)

## How to Run Locally

1. Clone this repository to your local machine.
2. Create a virtual environment and activate it.
3. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn tensorflow nltk matplotlib seaborn