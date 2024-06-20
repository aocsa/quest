import torch
import pyarrow as pa
import numpy as np
  
def strings_to_tensor(strings):
    # strings = ["hello", "world", "pytorch"]

    # Determine maximum string length
    max_length = max(len(s) for s in strings)

    # Map characters to integers (simple mapping: a=1, b=2, ..., z=26, space=0)
    # Note: This is a simplified example and might need adjustment for other use cases
    char_to_int = {chr(i): i - 96 for i in range(97, 123)}
    char_to_int[' '] = 0  # Adding space as 0 (assuming space might be used in other examples)

    # Initialize an empty tensor of shape (n, m) with zeros
    n = len(strings)
    max_length = max_length if max_length > 2 else 2
    m = max_length
    tensor = torch.zeros((n, m), dtype=torch.long)

    # Fill the tensor with integer codes for each character
    for i, string in enumerate(strings):
        tensor[i, :len(string)] = torch.tensor([char_to_int.get(char.lower(), 0) for char in string])
    return tensor

def tensor_to_strings(tensor):
    # Reverse mapping from integer to characters
    int_to_char = {i - 96: chr(i) for i in range(97, 123)}
    int_to_char[0] = ' '  # Assuming 0 was used for padding/spaces
        
    # Initialize an empty list to store the decoded strings
    strings = []

    # Convert each row in the tensor back to a string
    if tensor.dim() == 1:
        row = tensor 
        string = ''.join(int_to_char.get(int(char), '') for char in row if int(char) != 0)
        strings.append(string.strip())
    else:
        for row in tensor:
            string = ''.join(int_to_char.get(int(char), '') for char in row if int(char) != 0)
            strings.append(string.strip())
    return strings

