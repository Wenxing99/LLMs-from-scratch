import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    '''
    Greedy decoding, at each step, the model chooses the token with the highest probability as its next output
    '''
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        # the following code is used for inference, and telling torch does not save gradients, optimizer states, etc.
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # [B, T, vocab_size] -> [B, vocab_size]
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1) # [B, vocab_size]

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probs, dim=-1, keepdim=True) # [B, 1]

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # [B, T+1]

    return idx