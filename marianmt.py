from transformers import MarianMTModel, MarianTokenizer

def translate_file(input_file_path, output_file_path):
    model_name = "Helsinki-NLP/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = [text[i:i+400] for i in range(0, len(text), 400)]
    translations = []
    for sentence in sentences:
        tokens = tokenizer.prepare_seq2seq_batch([sentence], return_tensors="pt")
        translation = model.generate(**tokens)
        translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
        translations.append(translated_text)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(translations))

translate_file("input.txt", "translated_output.txt")

