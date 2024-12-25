# Standard library imports
import re
from urllib.parse import urlparse

# Third-party imports: Core Data Science & ML
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from dask_ml.model_selection import train_test_split

# Deep Learning frameworks
import tensorflow as tf
import keras
from keras.models import load_model
from transformers import AutoTokenizer, TFAutoModel

# NLP tools
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Text processing
import emoji

# Download NLP assets
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Preprocessing function definitions
def encode_labels(labels):
  manual_mapping = {
      'Politics': 0,
      'Sports': 1,
      'Media': 2,
      'Market & Economy': 3,
      'STEM': 4
  }
  encoded_labels = [manual_mapping[label] for label in labels]
  return encoded_labels

def extract_url_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    query = parsed_url.query
    tld = domain.split('.')[-1]
    domain = re.sub(r'\bwww\b', '', domain)
    domain = re.sub(r'\.com\b', '', domain)
    return  domain+' '+path

def place_holder_url(text):
    if not isinstance(text, str):
        return ""
    urls = re.findall(r'https?://(?:www\.)?[^\s/$.?#].[^\s]*', text)
    for url in urls:
        text = text.replace(url, extract_url_features(url))
    return text

def remove_urls(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'https?://(?:www\.)?[^\s/$.?#].[^\s]*', '', text)


def clean_text(text):
    if not isinstance(text, str):
        return " "
    text = text.lower()
    text = re.sub(r'(.)\1+', r'\1', text) #repetitive oooh ahhh
    text = re.sub(r'\s+', ' ', text)

    text_no_emojis = emoji.replace_emoji(text, replace=' ')
    text_no_numbers = re.sub(r'\d+', ' ', text_no_emojis)

    return text_no_numbers

def preprocess(text):
    text = place_holder_url(text)
    text = clean_text(text)
    # Tokenize using BERT tokenizer
    # and return input_ids instead of word embeddings
    return text

# File paths 
data_path = 'dataset/'
model_path = 'models/transformers/'
output_path = 'output/'

# Loading pretrained embedding model

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

transformer_model = TFAutoModel.from_pretrained('distilbert-base-uncased')

# Preprocessing text data
test = dd.read_csv(dataset_path + '/test.csv')
X_test=test['Discussion']
X_test = X_test.apply(preprocess, meta=('Discussion', 'object')).compute()
X_test=list(X_test)
encoded_X_test=tokenizer(
    X_test,
    add_special_tokens=True,
    max_length=128,
    padding="max_length",
    truncation=True,
    return_tensors="np"
)
X_test_input_ids = encoded_X_test["input_ids"]
X_test_attention_masks = encoded_X_test["attention_mask"]

batch_size = 32  
total_samples = X_test_input_ids.shape[0]
hidden_size = 768  
seq_len = X_test_input_ids.shape[1]


total_samples = X_test_input_ids.shape[0]
X_test_embeddings = np.zeros((total_samples, seq_len, hidden_size))

for start_idx in range(0, total_samples, batch_size):
    end_idx = min(start_idx + batch_size, total_samples)
    batch_input_ids = X_test_input_ids[start_idx:end_idx]
    batch_attention_masks = X_test_attention_masks[start_idx:end_idx]
    
    batch_output = transformer_model(batch_input_ids, attention_mask=batch_attention_masks).last_hidden_state
    
    X_test_embeddings[start_idx:end_idx] = batch_output.numpy()


# Loading our transformer based classifier
loaded_model = load_model(model_path + 'model-e24.keras')

y_test=loaded_model.predict(X_test_embeddings)
predictions=np.argmax(y_test, axis=1)

# Saving predictions to file
output=pd.DataFrame(predictions)
output.index +=1
output.index.name='SampleID'
output.columns = ['Discussion']
output.to_csv(output_path + 'transformer_outpur.csv', index=True)