from tokenizer import clean_and_tokenize
from classifier import NaiveBayesTextCore

if __name__ == "__main__":
    engine = NaiveBayesTextCore()
    
    # Feeding training metrics
    engine.train_entry(clean_and_tokenize("Urgent lottery reward winner code now"), "spam")
    engine.train_entry(clean_and_tokenize("Hey let us meet for project study tonight"), "ham")
    engine.train_entry(clean_and_tokenize("Claim crypto cash prize credit voucher"), "spam")
    engine.train_entry(clean_and_tokenize("Review the engineering draft guidelines"), "ham")
    
    query = "Urgent prize credit cash"
    tokenized_query = clean_and_tokenize(query)
    
    spam_score = engine.calculate_log_likelihood(tokenized_query, "spam")
    ham_score = engine.calculate_log_likelihood(tokenized_query, "ham")
    
    final_decision = "SPAM 🚨" if spam_score > ham_score else "HAM ✅"
    
    print("📬 BayesText-AI Probabilistic Inference Terminal...")
    print(f"📝 Raw Content Stream: \"{query}\"")
    print(f"🔮 Automated Pipeline Classification Verdict: [{final_decision}]")
