import torch
from transformers import MarianMTModel, MarianTokenizer

# torch.cuda.empty_cache()

en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    # .cuda()

fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    # .cuda()

src_text = [
    "Hi how are you?",
]

translated_tokens = en_fr_model.generate(
    **{k: v for k, v in en_fr_tokenizer(src_text, return_tensors="pt", padding=True, max_length=512).items()},
    # **{k: v.cuda() for k, v in en_fr_tokenizer(src_text, return_tensors="pt", padding=True, max_length=512).items()},
    do_sample=True,
    top_k=10,
    temperature=2.0,
)
in_fr = [en_fr_tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

bt_tokens = fr_en_model.generate(
    **{k: v for k, v in fr_en_tokenizer(in_fr, return_tensors="pt", padding=True, max_length=512).items()},
    # **{k: v.cuda() for k, v in fr_en_tokenizer(in_fr, return_tensors="pt", padding=True, max_length=512).items()},
    do_sample=True,
    top_k=10,
    temperature=2.0,
)
in_en = [fr_en_tokenizer.decode(t, skip_special_tokens=True) for t in bt_tokens]

print(in_en)