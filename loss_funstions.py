import torch

class OHEM(torch.nn.Module):                                                                                                                                                                             
    def __init__(self, loss_function = torch.nn.functional.binary_cross_entropy, ratio: float = 2/3, ) -> None:                                      
        self.loss_function = loss_function
        self.ratio = ratio                                                                                                                                
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, ratio: float = None) -> torch.Tensor:                                           
        if ratio is not None: self.ratio = ratio  
        samples2take = int(self.ratio * preds.shape[0])
        preds_ = preds.clone()
        losses = torch.autograd.Variable(torch.zeros(preds_.shape[0]))
        for i in range(preds.shape[0]):
            losses[i] = self.loss_function(preds_[i], targets[i])
        _, inds = torch.topk(losses, samples2take)
        return torch.mean(torch.index_select(losses, 0, inds))


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:   
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection)/(inputs.sum() + targets.sum())  
        return 1 - dice

def dice_bce_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss_val = dice_loss(preds, targets)
        bce_loss_val = torch.nn.functional.binary_cross_entropy(preds, targets)    
        return 0.5 * (dice_loss_val + bce_loss_val)