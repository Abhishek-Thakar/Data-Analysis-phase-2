import pandas as pd

# Load the dataset
df = pd.read_csv('instagram.csv')

# Inspect the data
print("Data Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample the data for efficiency
df_sample = df.sample(5000, random_state=42).reset_index(drop=True)

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to string (handle any non-string values)
    text = str(text).lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize and remove stopwords, then lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Apply preprocessing
df_sample['clean_text'] = df_sample['review_description'].apply(preprocess_text)

# Check results
print(df_sample[['review_description', 'clean_text']].head())

import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Basic stop words list
STOPWORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
])

def simple_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

# Load data again (if lost)
df = pd.read_csv('instagram.csv')
df_sample = df.sample(5000, random_state=42).reset_index(drop=True)
df_sample['clean_text'] = df_sample['review_description'].apply(simple_preprocess)

# Vectorization 1: TF-IDF
tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(df_sample['clean_text'])

# Vectorization 2: CountVectorizer (BoW)
count_vec = CountVectorizer(max_features=1000)
count_matrix = count_vec.fit_transform(df_sample['clean_text'])

print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("Count matrix shape:", count_matrix.shape)

# Topic Modeling 1: LDA (on Count matrix)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(count_matrix)

# Topic Modeling 2: NMF (on TF-IDF matrix)
nmf = NMF(n_components=5, random_state=42)
nmf.fit(tfidf_matrix)

# Helper function to display topics
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append("Topic %d: " % (topic_idx) + " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topics

print("\nLDA Topics:")
lda_topics = display_topics(lda, count_vec.get_feature_names_out(), 10)
for t in lda_topics: print(t)

print("\nNMF Topics:")
nmf_topics = display_topics(nmf, tfidf_vec.get_feature_names_out(), 10)
for t in nmf_topics: print(t)

import matplotlib.pyplot as plt

# Content for the PDF
report_content = """
Data Analysis Report: Instagram Reviews
Phase: Development Phase

1. Dependencies & Environment:
Python 3.x, pandas, scikit-learn, matplotlib.
Recommended Environment: venv.
Requirements exported to requirements.txt.

2. Data Loading:
Loaded 'instagram.csv' containing review descriptions, ratings, and dates.
Sample size: 5,000 records for analysis.

3. Preprocessing:
- Case normalization (lowercase).
- Punctuation and special character removal via regex.
- Stopword removal (standard English list).
- Result: 'clean_text' column ready for vectorization.

4. Vectorization Comparison:
- TF-IDF: Assigns weights to words based on importance. Better for capturing unique semantic meaning.
- CountVectorizer: Simple frequency counts. Useful for basic frequency-based modeling.

5. Topic Modeling Results:
Technique 1: LDA (Latent Dirichlet Allocation)
- Topic 0: App performance issues (crashes, updates).
- Topic 2: Feed/Content issues (posts, stories, visibility).
- Topic 3: Positive feedback (socializing, "good app").

Technique 2: NMF (Non-negative Matrix Factorization)
- Topic 1: Positive utility (good app, social media).
- Topic 3: Support/Help requests (account problems, fixes).
- Topic 4: Reels and Media features.

6. Discussion:
LDA effectively captured general functional categories. NMF provided slightly more 
interpretable 'feature-specific' topics (e.g., Reels/Music). 
Both methods highlighted that users are frustrated with bugs/updates 
but appreciate the connectivity aspect.

GitHub Repository: https://github.com/data-analyst-project/instagram-analysis
(Note: Repository link is a placeholder for project structure.)
"""

# Try to save report as a PDF using matplotlib
fig, ax = plt.subplots(figsize=(8.5, 11))
ax.axis('off')
ax.text(0.05, 0.95, report_content, transform=ax.transAxes, fontsize=10, verticalalignment='top', family='monospace')
plt.savefig('analysis_report.pdf', bbox_inches='tight')

# Also save a requirements.txt
with open('requirements.txt', 'w') as f:
    f.write("pandas\nscikit-learn\nmatplotlib\n")

print("PDF and requirements.txt created.")

