"""
knn_probe.py — k-NN embedding quality probe for protein language models.

Evaluates whether a trained model's embeddings cluster proteins by fold label,
using k-nearest-neighbor retrieval as the scoring mechanism. No labels are used
during embedding — only during scoring. This makes the probe a test of what
the model learned implicitly from sequence statistics alone.

Metric: hit rate — fraction of queries where at least one of the top-k neighbors
shares the query's fold label. Reported alongside a per-query hypergeometric
baseline (expected hit rate under random retrieval) and lift (hit_rate - baseline)
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from scipy.stats import hypergeom
import torch
from torch.utils.data import DataLoader, Dataset"""

def score_knn(embeddings, labels, k=10, min_fold_size=2):
    """Compute hit rate, baseline, and lift for k-NN on embeddings.
    Args:
        embeddings:     torch.Tensor of shape n_samples, embedding_dim
        labels:         list of length n_samples, containing the fold labels
        k:              number of nearest neighbors to consider
        min_fold_size:  minimum number of members in a fold for it to be considered
    Returns:
        hit_rate:       fraction of queries where at least one of the top-k neighbors shares the fold label
        baseline:       expected hit rate under hypergeometric distribution
        lift:           hit_rate - baseline
    """
    counts = Counter(labels)
    usable_folds = {f for f, c in counts.items() if c >= min_fold_size}
    mask = [label in usable_folds for label in labels]
    embeddings_np = embeddings.numpy()
    query_embeddings = embeddings_np[mask]
    
    
    usable_indices = [i for i, m in enumerate(mask) if m]
    
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine")
    nn.fit(embeddings_np)
    distances, indices = nn.kneighbors(query_embeddings)
    hit_rate = 0.0
    baseline_av = 0.0
    for i in range(len(query_embeddings)):
        query_label = labels[usable_indices[i]]
        neighbor_labels = [labels[j] for j in indices[i] if j != usable_indices[i]]
        hit = any(nl == query_label for nl in neighbor_labels)
        baseline_prob = hypergeom.sf(0, len(labels)-1, counts[query_label]-1, k)
        baseline_av += baseline_prob
        hit_rate += hit
        

    hit_rate = hit_rate / len(query_embeddings)
    baseline = baseline_av / len(query_embeddings)
    lift = hit_rate - baseline
    return hit_rate, baseline, lift 
    

class SequenceDataset(Dataset):
    """Dataset wrapping a list of protein sequences for batched embedding.

Tokenizes and truncates sequences at construction time via __getitem__.
Designed for use with embed_sequences — not for training.
"""
    def __init__(self, sequences, tokenizer, max_len):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # tokenize and truncate — returns token ids as a tensor
        token_ids = self.tokenizer.encode(self.sequences[idx])[:self.max_len]
        return torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch):
    # pad sequences in batch to the same length
    max_len = max(len(x) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
        attention_mask[i, :len(x)] = 1
    return padded, attention_mask


def embed_sequences(model, sequences, tokenizer, device, batch_size, max_len):
    """
    Embed a list of protein sequences using mean pooling over residue hidden states.
    
    Args:
        model:      trained PLM, in eval mode
        sequences:  list of raw amino acid strings
        tokenizer:  tokenizer matching the model's vocabulary
        device:     torch.device
        batch_size: number of sequences per forward pass
        max_len:    truncate sequences longer than this
    Returns:
        torch.Tensor of shape (n_samples, d_model), on CPU
    """
    dataset = SequenceDataset(sequences, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # hidden_states: (batch, seq_len, d_model)
            output = model(input_ids, return_hidden_states=True)
            hidden_states = output["hidden_states"]  # [B, L, D]

            # mean pool over residue dimension, ignoring padding
            # expand mask to (batch, seq_len, 1) for broadcasting
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            lengths = mask.sum(dim=1)
            embeddings = summed / lengths  # (batch, d_model)

            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)  # (n_samples, d_model)