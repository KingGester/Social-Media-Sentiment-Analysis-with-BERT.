def tokenize_function(examples, tokenizer=None):
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided.")
    return tokenizer(examples["Text"], padding="max_length", truncation=True)