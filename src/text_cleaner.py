from typing import Any
import re
import string
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('stopwords')

class TextCleaner:
    def __init__(self):
        # Load stopwords once and reuse
        self.stop_words = set(stopwords.words('english'))

    def __call__(self, text: str) -> str:
        text = self.remove_emojis_and_unrecognizable_characters(text)
        text = self.remove_punctuations(text)
        # text = self.remove_stopwords(text)
        return text

    def remove_emojis_and_unrecognizable_characters(self, text: str) -> str:

        date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\+?\d{1,3}\s?(\(\d{1,3}\)\s?|\d{1,3}[-.\s]?)\d{1,3}[-.\s]?\d{1,4}([-.\s]?\d{1,4})?'

        # Remove every date 
        text = re.sub(date_pattern,'',text)

        # Remove email patterm
        text = re.sub(email_pattern,'',text)

        # Remove phone patterm
        text = re.sub(phone_pattern,'',text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]*>', '', text)
        # Remove URLs   
        text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|www\.\S+', '', text)

        # Remove special characters and multiple spaces
        text = re.sub(r'[^\w\s]', '', text)
         
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_punctuations(self, text: str) -> str:
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)

    def remove_stopwords(self, text: str) -> str:
        text = text.lower().split()
        filtered_words = [word for word in text if word not in self.stop_words]
        return ' '.join(filtered_words)

