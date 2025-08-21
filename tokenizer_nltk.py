import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize

# Example text
text = """Hello! My name is Hiba. I am doing bachelors in Data Science, and I love AI. 
Do you like programming? If yes then connect with me we'll explore more on AI together. """

# Sentence Tokenization
sentences = sent_tokenize(text)
print("✅ Sentences:")
for s in sentences:
    print("-", s)

# Word Tokenization
print("\n✅ Words:")
for s in sentences:
    words = word_tokenize(s)
    print(words)
