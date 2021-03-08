import numpy as np


def get_vocab(words):
    vocab = {}
    vocab_list = []
    index = 0
    for word in words:
        for type_ in word:
            if type_ not in vocab:
                vocab[type_] = index
                vocab_list.append(type_)
                index += 1

    return vocab, vocab_list


def load_data(filename="A5-data.txt"):
    with open(filename, "r") as f:
        words = f.read().split()

    for i, word in enumerate(words):
        words[i] = list(word) + ["<s>"]

    vocab, vocab_list = get_vocab(words)

    return words, vocab, vocab_list


def count_tokens_and_bigrams(words, vocab, vocab_list):
    vocab_size = len(vocab_list)
    # counts[i, j] = prob of bigram (i,j)
    counts = np.zeros((vocab_size, vocab_size), dtype=int)
    seq_len = 0

    for word in words:
        seq_len += len(word)
        # Iterate over bigrams in this word.
        for i in range(1, len(word)):
            # Add a count for this bigram.
            counts[vocab[word[i - 1]], vocab[word[i]]] += 1

    return counts, seq_len


def merge_bigram(words, type1, type2):
    for word in words:
        # Iterate over bigrams in this word.
        i = 1
        while i < len(word):
            # Add a count for this bigram.
            if type1 == word[i - 1] and type2 == word[i]:
                # Merge the two.
                word[i - 1] = word[i - 1] + word[i]
                word.pop(i)
            i += 1

    return words


def train(words, vocab, vocab_list, target_vocab_size=np.inf):

    counts, _ = count_tokens_and_bigrams(words, vocab, vocab_list)

    vocab_sizes = []
    seq_lens = []
    merges = []
    converged = False  # Whether max bigram count is one.

    while (not converged) and (len(vocab_list) < target_vocab_size):

        # Merge.
        max1, max2 = np.unravel_index(np.argmax(counts), counts.shape)
        type1 = vocab_list[max1]
        type2 = vocab_list[max2]
        words = merge_bigram(words, type1, type2)

        # Update vocabulary with new bigrams.
        vocab, vocab_list = get_vocab(words)

        # Reset counts of types.
        counts, seq_len = count_tokens_and_bigrams(words, vocab, vocab_list)
        converged = np.max(counts) == 1

        vocab_sizes.append(len(vocab_list))
        seq_lens.append(seq_len)
        merges.append((type1, type2))

    return words, vocab_sizes, seq_lens, merges


def encode(word, merges):
    words = [list(word) + ["<s>"]]
    for type1, type2 in merges:
        words = merge_bigram(words, type1, type2)

    return words[0]

