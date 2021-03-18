import torch 
from torch.nn import functional as F

def signal_game_loss(_sender_input,  _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == _labels).detach().float()
    loss = F.cross_entropy(receiver_output, _labels, reduction="none")
    return loss, {'acc': acc}

class ImageClasLoss:
    """
    This class is a simple wrapper of the multilabel image classification loss function. We use it as a workaround to be able to specify
    the image classification loss weight from command line because we cannot add another parameter to the loss 
    function to be compatible with the EGG library.
    """
    def __init__(self, ic_loss_weight, num_imgs):
        super().__init__()
        self.ic_loss_weight = ic_loss_weight
        self.num_imgs = num_imgs

    def get_loss(self, _sender_input,  _message, _receiver_input, receiver_output, _labels):
        sg_receiver_out = receiver_output[:, :self.num_imgs]
        sg_labels = _labels[:,:1].long().squeeze()
        sg_acc = (sg_receiver_out.argmax(dim=1) == sg_labels).detach().float()
        sg_loss = F.cross_entropy(sg_receiver_out, sg_labels, reduction="none").unsqueeze(dim=1)
        
        ic_receiver_out = receiver_output[:, self.num_imgs:]
        ic_labels = _labels[:,1:]
        ic_loss = F.binary_cross_entropy_with_logits(ic_receiver_out, ic_labels, reduction="none")
        """
        Average the loss over number of images - without doing that, in games with larger number of images
        the image classification loss would automatically have larger impact on the total loss
        compared to the standard signalling game loss which is not desirable 
        (we want the impact to be only on the specified weight (ic_loss_weight))
        """
        ic_loss = ic_loss.mean(dim=1, keepdim=True)
        
        loss = torch.cat((sg_loss, self.ic_loss_weight*ic_loss), dim=1)
        loss = torch.mean(loss, dim=1)
        
        """
        Note/FIXME: If we pass the loss in the first returned variable, the printed loss won't be the same as it's mean passed as a value in the
        dictionary returned below. I think this is due to some masking, see: 
        https://github.com/facebookresearch/EGG/blob/3a429c27b798a24b12b05486f8832d4a82ea3327/egg/core/gs_wrappers.py#L501
        """
        
        return loss, {# "sg_loss":sg_loss.mean(), 
                      #"ic_loss":ic_loss.mean(), 
                      "acc": sg_acc,
                      "img_class_acc":multilabel_acc(ic_receiver_out, ic_labels)}
        
def multilabel_acc(preds, labels, thresh = 0.5):
    """
    Based on Section 7.1.1 in https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.364.5612&rep=rep1&type=pdf.
    """
    # pass receiver outputs through a sigmoid
    pred_logits = torch.sigmoid(preds)
    # convert probabilities into labels based on an arbitrary threshold
    pred_labels = (pred_logits>thresh).long()
    return (pred_labels == labels).float().mean(dim=1).detach()
