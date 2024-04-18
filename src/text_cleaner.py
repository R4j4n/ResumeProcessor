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
        # Remove HTML tags
        text = re.sub(r'<[^>]*>', '', text)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
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

# Usage example
cleaner = TextCleaner()
clean_text = cleaner("• Your example resume text here with URLs, emojis, •••••••• and other characters.😂😂😂😂😂😂😂😂😂")
print(clean_text)
