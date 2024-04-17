from typing import Any
import re
import string
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')


# Text cleaner added
class TextCleaner:

    def __call__(self,text) -> Any:
        emojis_cleared = self.remove_emojis_and_unrecognizable_characters(text)

        punctuations_removed = self.remove_punctuations(emojis_cleared)

        cleaned_text = self.remove_stopwords(punctuations_removed)
        return cleaned_text
    
    def remove_emojis_and_unrecognizable_characters(self,text):
        article = re.sub(r'\([^)]*\)', r'', text)
        article = re.sub(r'\[[^\]]*\]', r'', article)
        article = re.sub(r'<[^>]*>', r'', article)
        article = re.sub(r'^https?:\/\/.[\r\n]', '', article)
        article = re.sub('http[s]?://\S+', '', article)
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

    def remove_stopwords(self,text):
        text = text.lower()
        words = text.split()
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        text = ' '.join(words)
        return text


words = "John Doe is an experienced software engineer with a strong background in developing scalable web applications. His technical skills include proficiency in JavaScript, Python, and Java, as well as a deep understanding of React and Node.js frameworks. John has worked on various projects that required collaborative efforts and has consistently delivered high-quality code. He is also familiar with agile methodologies and has participated in numerous sprint planning sessions and daily stand-ups. In addition to his technical abilities, John has demonstrated strong problem-solving skills and the ability to adapt to new technologies quickly. He is a team player who is always eager to learn and contribute to project success."
cleaner = TextCleaner()

print(f"The cleaned text is as follows:\n\n\t{cleaner(words)}")
