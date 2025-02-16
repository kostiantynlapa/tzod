from collections import Counter
import re

def most_common_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    
    words = re.findall(r'\b\w+\b', text)
    word_counts = Counter(words)
    
    return word_counts.most_common(5)

# Використання
file_path = "file.txt"
print(most_common_words(file_path))
