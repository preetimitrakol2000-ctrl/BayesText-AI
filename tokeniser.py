def clean_and_tokenize(text_blob):
    """Normalizes case attributes and splits text into atomic word arrays."""
    punctuations = ".,!?;:"
    cleaned = text_blob.lower()
    for p in punctuations:
        cleaned = cleaned.replace(p, "")
    return [word for word in cleaned.split(" ") if len(word) > 2]
