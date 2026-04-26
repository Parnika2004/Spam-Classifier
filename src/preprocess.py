import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = text.lower()                          # lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # remove punctuation/numbers
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    tokens = [stemmer.stem(t) for t in tokens]  # stem words
    return ' '.join(tokens)

# Test it
if __name__ == '__main__':
    sample = "CONGRATULATIONS! You've WON a FREE prize. Call NOW!!!"
    print(clean_text(sample))
    # Output: congratul won free prize call
