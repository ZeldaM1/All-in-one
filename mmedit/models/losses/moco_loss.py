import torch
from torch import nn
import torch.nn.functional as F
from ..registry import LOSSES

@LOSSES.register_module()
class MocoLoss(nn.Module):
    def __init__(self,model_paths_moco="/output/v-huiminzeng/moco_v2_800ep_pretrain.pt",loss_weight=1.0):
        super(MocoLoss, self).__init__()
        self.loss_weight=loss_weight
        print("Loading MOCO model from path: {}".format(model_paths_moco))
        self.model = self.__load_model(model_paths_moco)
        # self.model.cuda()
        self.model.eval()

    @staticmethod
    def __load_model(model_paths_moco_path):
        import torchvision.models as models
        model = models.__dict__["resnet50"]()
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        checkpoint = torch.load(model_paths_moco_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        # remove output layer
        model = nn.Sequential(*list(model.children())[:-1]).cuda()
        return model

    def extract_feats(self, x):
        x = F.interpolate(x, size=224)
        x_feats = self.model(x)
        x_feats = nn.functional.normalize(x_feats, dim=1)
        x_feats = x_feats.squeeze()
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = y_hat.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)
        x_feats = x_feats.repeat(n_samples,1)
        y_feats = y_feats.repeat(n_samples,1)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        sim_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            sim_logs.append({'diff_target': float(diff_target),
                             'diff_input': float(diff_input),
                             'diff_views': float(diff_views)})
            loss += 1 - diff_target
            sim_diff = float(diff_target) - float(diff_views)
            sim_improvement += sim_diff
            count += 1

        return  (loss / count)*self.loss_weight, sim_improvement / count, sim_logs


    
@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
 
        self.loss_weight = loss_weight
        self.entropy = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * self.entropy(pred,target)

