import torch


class BCE_OHEM(torch.nn.BCELoss):                                                                                                                                                                             
    def __init__(self, ratio: float = 2/3) -> None:      
        super(BCE_OHEM, self).__init__(None, True)                                 
        self.ratio = ratio                                                         
                                                                                   
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, ratio: float = None) -> torch.Tensor:                                           
        if ratio is not None: self.ratio = ratio  
        samples2take = int(self.ratio * preds.shape[0])
        losses = torch.Tensor(list(map(lambda x: torch.nn.functional.binary_cross_entropy(x[0], x[1], reduce='sum'), zip(preds, targets))))
        _, inds = torch.topk(losses, samples2take)
        return torch.mean(torch.index_select(losses, 0, inds))


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):

        prediction_shape = list(preds.shape)
        prediction = torch.reshape(preds >= 0.5, [prediction_shape[0], torch.prod(torch.Tensor(prediction_shape[1:]), dtype=torch.int32)])
    
        mask_shape = list(targets.shape)
        mask = torch.reshape(mask >= 0.5, [mask_shape[0], torch.prod(torch.Tensor(mask_shape[1:]), dtype=torch.int32)])
        intersection_mask = torch.logical_and(prediction, mask) 

        intersection = torch.sum(intersection_mask.type(torch.FloatTensor))
        total_sum = (prediction + mask).type(torch.FloatTensor)
        dice = (2. * intersection) / total_sum
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):

        prediction_shape = list(preds.shape)
        prediction = torch.reshape(preds >= 0.5, [prediction_shape[0], torch.prod(torch.Tensor(prediction_shape[1:]), dtype=torch.int32)])
    
        mask_shape = list(targets.shape)
        mask = torch.reshape(mask >= 0.5, [mask_shape[0], torch.prod(torch.Tensor(mask_shape[1:]), dtype=torch.int32)])
        intersection_mask = torch.logical_and(prediction, mask) 
           
        intersection = torch.sum(intersection_mask.type(torch.FloatTensor))
        total_sum = (prediction + mask).type(torch.FloatTensor)
        dice = (2. * intersection) / total_sum
        
        return 1 - dice