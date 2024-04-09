from typing import Any
import re
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


nltk.download('stopwords')


# Text cleaner added
class TextCleaner:

    def __call__(self,text) -> Any:
        emojis_cleared = self.remove_emojis_and_unrecognizable_characters(text)

        punctuations_removed = self.remove_punctuations(emojis_cleared)

        stemmed_text = self.stemming(punctuations_removed)

        return stemmed_text
    
    def remove_emojis_and_unrecognizable_characters(self,text):
        article = re.sub(r'\([^)]*\)', r'', text)
        article = re.sub(r'\[[^\]]*\]', r'', article)
        article = re.sub(r'<[^>]*>', r'', article)
        article = re.sub(r'^https?:\/\/.[\r\n]', '', article)
        article = re.sub('http[s]?://\S+', '', article)
        article = re.sub(r'[a-zA-Z0-9]+', '', article)
        article = re.sub('[!+*-@,#%(&$_?.^]', '', article)
        article = re.sub(' +', ' ',article)
        article = article.replace(u'\ufeff','')
        article = article.replace(u'\xa0', u' ')
        article = article.replace('  ', ' ')
        article = article.replace(' , ', ', ')

        return article

    def remove_punctuations(self, text):
        result = ""
        for character in text:
            result += "" if character in string.punctuation else character
        return result

    def stemming(self,text):
        text = text.lower()
        words = text.split()
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]

        text = ' '.join(words)

        return text
