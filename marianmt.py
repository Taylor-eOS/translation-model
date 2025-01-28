from transformers import MarianMTModel, MarianTokenizer

# Load the MarianMT model and tokenizer for German-to-English translation
def load_model():
    model_name = "Helsinki-NLP/opus-mt-de-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Split a long text into smaller overlapping chunks for translation
def split_into_chunks(text, tokenizer, max_length=512, overlap=50):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start = end - overlap  # Create overlap for continuity
    return chunks

# Translate a single chunk of tokens
def translate_chunk(model, tokenizer, chunk):
    inputs = {"input_ids": chunk.unsqueeze(0)}
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Translate a long text by splitting it into chunks and reassembling the translation
def translate_long_text(model, tokenizer, text, max_length=512, overlap=50):
    chunks = split_into_chunks(text, tokenizer, max_length=max_length, overlap=overlap)
    translated_chunks = [translate_chunk(model, tokenizer, chunk) for chunk in chunks]
    return " ".join(translated_chunks)

# Process a file line by line, handling long sentences with smart splitting
def translate_file(input_file, output_file, model, tokenizer, max_length=512, overlap=50):
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            lines = infile.readlines()

        translated_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            translated_text = translate_long_text(model, tokenizer, line, max_length=max_length, overlap=overlap)
            translated_lines.append(translated_text)
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write("\n".join(translated_lines))
        print(f"Translation complete! Results saved in '{output_file}'")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main function
def main():
    # Specify input and output files
    input_file = "input.txt"
    output_file = "output.txt"
    print("Loading model...")
    model, tokenizer = load_model()
    print(f"Translating text from '{input_file}' to '{output_file}'...")
    translate_file(input_file, output_file, model, tokenizer)

if __name__ == "__main__":
    main()

