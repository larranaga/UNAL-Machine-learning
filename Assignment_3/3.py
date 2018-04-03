import unicodedata

with open('spanish.txt') as f:
    data = ''.join(i for i in f.read() if not i.isdigit())
    d = ''.join(c for c in unicodedata.normalize('NFKD', data))
    spanish_data = []
    for word in d.split():
        if word.find('Ã') == -1:
            #print(word)
            spanish_data.append(word)
print(spanish_data)
