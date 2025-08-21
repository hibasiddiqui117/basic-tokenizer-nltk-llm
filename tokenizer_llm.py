from transformers import AutoTokenizer

def llm_tokenizer(text):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Encode text -> IDs
    encoded = tokenizer.encode(text)

    # Convert IDs back to tokens
    tokens = tokenizer.convert_ids_to_tokens(encoded)

    # Decode IDs -> text
    decoded = tokenizer.decode(encoded)

    return tokens, encoded, decoded

if __name__ == "__main__":
    with open("examples.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("📌 Original Text:\n", text, "\n")

    tokens, encoded, decoded = llm_tokenizer(text)

    print("✅ Tokens:", tokens)
    print("✅ Encoded IDs:", encoded)
    print("✅ Decoded text:", decoded)
