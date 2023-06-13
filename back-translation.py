import torch
import pandas as pd
import numpy as np
from transformers import MarianMTModel, MarianTokenizer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to(device)
    # .cuda()

fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en").to(device)
    # .cuda()

src_text = [
    "Hi how are you?",
]

def back_translate(text):

    translated_tokens = en_fr_model.generate(
        **{k: v.to(device) for k, v in en_fr_tokenizer(text, return_tensors="pt", padding='max_length', max_length=512).items()},
        # **{k: v.cuda() for k, v in en_fr_tokenizer(text, return_tensors="pt", padding=True, max_length=512).items()},
        do_sample=True,
        top_k=10,
        temperature=2.0,
    )
    in_fr = [en_fr_tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

    bt_tokens = fr_en_model.generate(
        **{k: v.to(device) for k, v in fr_en_tokenizer(in_fr, return_tensors="pt", padding='max_length', max_length=512).items()},
        # **{k: v.cuda() for k, v in fr_en_tokenizer(in_fr, return_tensors="pt", padding=True, max_length=512).items()},
        do_sample=True,
        top_k=10,
        temperature=2.0,
    )

    in_en = [fr_en_tokenizer.decode(t, skip_special_tokens=True) for t in bt_tokens]

    return in_en

unsup = pd.read_csv('data/unsup/unsup_20000.csv')
print(unsup.head())

print(unsup['text'].apply(lambda x: len(x.split(' '))).value_counts())
print(back_translate(src_text))

newdf = pd.DataFrame(np.repeat(unsup.values, 3, axis=0))
newdf.columns = unsup.columns
print(newdf.head(10))

for i in range(len(newdf)):
    if i % 100 == 0:
        print(i)
    ori = newdf['text'][i][:512]
    aug = back_translate(ori)
    newdf.at[i, 'text'] = aug[0]


newdf.to_csv('data/unsup/unsup_20000_aug3.csv', index=False)

# print(in_en)