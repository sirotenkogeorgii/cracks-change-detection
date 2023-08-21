import torch


class BCE_OHEM(torch.nn.BCELoss):                                                                                                                                                                             
    def __init__(self, ratio: float = 2/3) -> None:      
        super(BCE_OHEM, self).__init__(None, True)                                 
        self.ratio = ratio                                                         
                                                                                   
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, ratio: float = None) -> torch.Tensor:                                           
        if ratio is not None: self.ratio = ratio  
        samples2take = int(self.ratio * preds.shape[0])
        losses = torch.Tensor(list(map(lambda x: torch.nn.functional.binary_cross_entropy(x[0], x[1], reduce='sum'), zip(preds, targets))))
        return torch.mean(torch.topk(losses, samples2take).values)

