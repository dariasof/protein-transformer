def evaluate(
    model: ProteinMLM,
    val_loader: DataLoader,
    device: str,
) -> float:
    """Run eval loop, return perplexity on val set.
    Args:
        model:      Trained ProteinMLM instance.
        val_loader: DataLoader over the held-out validation split.
                    Should use shuffle=False.
        device:     'cuda' or 'cpu'.

    Returns:
        Perplexity as a float
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            output = model(input_ids, labels=labels)
            loss = output["loss"]
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    perplexity = math.exp(avg_loss)
    return perplexity
