import torch
import numpy as np
import torch.nn.functional as F

def memorization_nll(predictions, ground_truth):
    '''Returns average negative log-likelihood between generated logits and true continuation

    Attributes:
        ground_truth: tuple of pytorch tensors of shape
            (batch_size,token_probabilites) and of length num_tokens,
            the shape of scores of generate method
        predictions: array of shape (batch_size,num_tokens)

    Returns:
        numpy array of shape (batch_size), the average likelihood
    '''

    token_wise_losses = []
    token_size = ground_truth.shape[1]
    for i in range(token_size):
        curr_loss = F.nll_loss(
            predictions[i].float(), 
            ground_truth[:, i].type(torch.LongTensor),
            reduction='none'
        )
        token_wise_losses.append(curr_loss.cpu().numpy().tolist())

    total_loss = np.asarray(token_wise_losses).transpose()
    total_loss = np.average(total_loss, axis=-1)
    return total_loss


def memorization_acc(predictions, ground_truth):
    '''Returns average accuracy between generated logits and true continuation

    Attributes:
        ground_truth: array of shape (batch_size, num_tokens), true continuation tokens
        predictions: array of shape (batch_size, num_tokens), generated context tokens

    Returns:
        numpy array of shape (batch_size), the average accuracy
    '''
    num_eq = torch.mean((predictions == ground_truth).float(), dim=-1)
    return num_eq.cpu().numpy()

if __name__ == '__main__':
    predictions = torch.ones(size=(32,32,500)).cuda() # vocab_size=500,batch_size=64,out_num_tokens=10
    ground_truth = torch.ones(size=(32,32),dtype=torch.int32).cuda() # out_batch_size,out_num_tokens
    predicted_tokens = torch.argmax(predictions, dim = -1).T + 1
    print(memorization_acc(ground_truth, predicted_tokens)) # Output: (64,)
    print(memorization_nll(predictions,ground_truth).shape) # Output: (64,)