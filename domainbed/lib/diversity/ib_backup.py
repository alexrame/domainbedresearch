# def get_negative_expectation(
#     q_samples
# ):
#     """Computes the negative part of a divergence / difference for dvkl
#     Args:
#         q_samples: Negative samples.
#     Returns:
#         torch.Tensor
#     """
#     return torch.log(torch.mean(q_samples))

# def permute_tensor_offset(t, offset=None):
#     len_t = t.size(0)
#     init_index = torch.arange(len_t)
#     if offset is None:
#         offset = random.randint(1, len_t - 1)

#     index = torch.fmod(init_index + offset, len_t)
#     return t[index]
