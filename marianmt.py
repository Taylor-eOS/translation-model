from transformers import MarianMTModel, MarianTokenizer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

def translate_file(input_file_path, output_file_path):
    model_name = "Helsinki-NLP/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Read the input file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Translate sentences in batches
    translations = []
    batch_size = 10
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**tokens)
        translated_batch = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        translations.extend(translated_batch)
    
    # Write translated text to output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(translations))

translate_file("input.txt", "translated_output.txt")

