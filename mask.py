import torch


# output:
# [[0, -∞ , -∞],
#  [0, 0 , -∞],
#  [0, 0 , 0]]
def generate_square_subsequent_mask(sz, device: str):
    # create a triangular matrix
    # and convert 1 to 0 and 0 to -∞
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_all_masks(source, target, device: str, padding_index: int):
    source_seq_len = source.shape[1]
    target_seq_len = target.shape[1]

    source_mask, target_mask = create_mask(source_seq_len, target_seq_len, device)
    source_padding_mask, target_padding_mask = create_padding_mask(source, target, padding_index)
    return source_mask, target_mask, source_padding_mask, target_padding_mask


# only need shape to create mask
def create_mask(source_seq_len, target_seq_len, device: str):
    # no mask for source
    source_mask = torch.zeros((source_seq_len, source_seq_len),
                              device=device).type(torch.bool)
    # generate the square mask for target
    # to hidden the future tokens
    # [[0, -∞ , -∞],
    #  [0, 0 , -∞],
    #  [0, 0 , 0]]
    target_mask = generate_square_subsequent_mask(target_seq_len, device)
    return source_mask, target_mask


# Just mark the padding index as True
def create_padding_mask(source, target, padding_index: int):
    source_padding_mask = source == padding_index
    target_padding_mask = target == padding_index
    return source_padding_mask, target_padding_mask
