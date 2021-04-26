import string

def normalizeString(s):
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude)
        s = s.lower()
        return s