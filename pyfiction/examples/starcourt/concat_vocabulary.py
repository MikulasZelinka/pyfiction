with open('vocabulary1.txt') as f:
    words1 = f.readlines()

with open('vocabulary2.txt') as f:
    words2 = f.readlines()

words = list(set(words1 + words2))

with open('vocabulary.txt', 'w') as f:
    f.writelines(words)
