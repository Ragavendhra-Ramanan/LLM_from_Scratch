import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total Number of character:", len(raw_text))

preprocessed = re.split(r'([,.?!"()\']|--|\s)',raw_text)
preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
all_words = set(preprocessed)
vocab_size = len(all_words)


all_tokens = (list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens) }
vocab_size = len(all_tokens)


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items() }

    def encode(self, text):
        preprocessed_text = re.split(r'([,.?!"()\']|--|\s)',text)
        preprocessed_text = [
            item.strip() for item in preprocessed_text if item.strip()
        ]
        preprocessed_text = [ text if text in self.str_to_int else "<|unk|>" for text in preprocessed_text]
        ids = [self.str_to_int[i] for i in preprocessed_text]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text
    
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "<|endoftext|> ".join((text1,text2))

tokenizer  = SimpleTokenizerV1(vocab=vocab)
print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))



