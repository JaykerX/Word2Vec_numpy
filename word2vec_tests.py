import numpy as np

data = np.load("word2vec_model_big.npz", allow_pickle=True)
W = data["W"]
W_out = data["W_out"]
vocab = data["vocab"].item()
id2word = data["id2word"].item()

print("Model loaded successfully")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def most_similar(word, topk=5):
    if word not in vocab:
        return []
    w_vec = W[vocab[word]]
    sims = []
    for i in range(len(W)):
        sims.append((id2word[i], cosine_similarity(w_vec, W[i])))
    sims.sort(key=lambda x: -x[1])
    return sims[1:topk+1]


if __name__ == "__main__":

    words_to_check = ["king", "queen", "man", "woman", "dog", "cat", "love", "work", "do", "great"]

    print("\nMost similar words:")
    for w in words_to_check:
        if w in vocab:
            sim_words = most_similar(w, topk=3)
            print(f"{w}: {[word for word, _ in sim_words]}")
        else:
            print(f"{w}: not in vocab")

    word_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("dog", "cat"),
        ("love", "hate"),
        ("work", "do"),
        ("great", "excellent"),
        ("king", "man"),
        ("queen", "woman"),
        ("dog", "wolf"),
        ("love", "affection")
    ]

    print("\nCosine similarities for word pairs:")
    for w1, w2 in word_pairs:
        if w1 in vocab and w2 in vocab:
            sim = cosine_similarity(W[vocab[w1]], W[vocab[w2]])
            print(f"Similarity({w1}, {w2}) = {sim:.4f}")
        else:
            print(f"Similarity({w1}, {w2}) = N/A (word not in vocab)")
