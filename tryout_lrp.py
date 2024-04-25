import torch
import torch.nn.functional as F


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))


def module_tracker(fwd_hook_func):
    def hook_wrapper(relevance_propagator_instance, layer, *args):
        relevance_propagator_instance.module_list.append(layer)
        return fwd_hook_func(relevance_propagator_instance, layer, *args)
    return hook_wrapper


class RelevancePropagator:
    def __init__(self, lrp_exponent, epsilon):
        self.p = lrp_exponent
        self.eps = epsilon
        self.module_list = []

    def compute_propagated_relevance(self, layer, relevance):
        if isinstance(layer, torch.nn.MaxPool2d):
            return self.max_pool_nd_inverse(layer, relevance)
        elif isinstance(layer, torch.nn.Conv2d):
            return self.conv_nd_inverse(layer, relevance)
        elif isinstance(layer, torch.nn.Linear):
            return self.linear_inverse(layer, relevance)
        return relevance

    def get_layer_fwd_hook(self, layer):
        if isinstance(layer, torch.nn.MaxPool2d):
            return self.max_pool_nd_fwd_hook
        if isinstance(layer, torch.nn.Conv2d):
            return self.conv_nd_fwd_hook
        if isinstance(layer, torch.nn.Linear):
            return self.linear_fwd_hook
        return self.silent_pass

    @module_tracker
    def silent_pass(self, m, in_tensor: torch.Tensor,
                    out_tensor: torch.Tensor):
        pass

    def linear_inverse(self, m, relevance_in):
        m.in_tensor = m.in_tensor.pow(self.p)
        w = m.weight.pow(self.p)
        norm = F.linear(m.in_tensor, w, bias=None)

        norm += torch.sign(norm) * self.eps
        relevance_in[norm == 0] = 0
        norm[norm == 0] = 1
        relevance_out = F.linear(relevance_in / norm,
                                 w.t(), bias=None)
        relevance_out *= m.in_tensor
        del m.in_tensor, norm, w, relevance_in
        return relevance_out

    @module_tracker
    def linear_fwd_hook(self, m, in_tensor: torch.Tensor,
                        out_tensor: torch.Tensor):
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "out_shape", out_tensor.size())

    def max_pool_nd_inverse(self, layer_instance, relevance_in):
        relevance_in = relevance_in.view(layer_instance.out_shape)
        invert_pool = F.max_unpool2d
        inverted = invert_pool(relevance_in, layer_instance.indices,
                               layer_instance.kernel_size, layer_instance.stride,
                               layer_instance.padding, output_size=layer_instance.in_shape)
        del layer_instance.indices
        return inverted

    @module_tracker
    def max_pool_nd_fwd_hook(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):
        # Save the return indices value to make sure
        tmp_return_indices = bool(m.return_indices)
        m.return_indices = True
        _, indices = m.forward(in_tensor[0])
        m.return_indices = tmp_return_indices
        setattr(m, "indices", indices)
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_shape', in_tensor[0].size())

    def conv_nd_inverse(self, m, relevance_in):
        relevance_in = relevance_in.view(m.out_shape)
        inv_conv_nd = F.conv_transpose2d
        conv_nd = F.conv2d
        with torch.no_grad():
            m.in_tensor = m.in_tensor.pow(self.p)
            w = m.weight.pow(self.p)
            norm = conv_nd(m.in_tensor, weight=w, bias=None,
                           stride=m.stride, padding=m.padding,
                           groups=m.groups)
            norm += torch.sign(norm) * self.eps
            relevance_in[norm == 0] = 0
            norm[norm == 0] = 1
            relevance_out = inv_conv_nd(relevance_in / norm,
                                        weight=w, bias=None,
                                        padding=m.padding, stride=m.stride,
                                        groups=m.groups)
            relevance_out *= m.in_tensor
            del m.in_tensor, norm, w
            return relevance_out

    @module_tracker
    def conv_nd_fwd_hook(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', out_tensor.size())


class InnvestigateModel(torch.nn.Module):
    def __init__(self, the_model, lrp_exponent=1, epsilon=1e-6):
        super(InnvestigateModel, self).__init__()
        self.model = the_model
        self.inverter = RelevancePropagator(lrp_exponent=lrp_exponent,
                                            epsilon=epsilon)
        self.register_hooks(self.model)

    def register_hooks(self, parent_module):
        for mod in parent_module.children():
            if list(mod.children()):
                self.register_hooks(mod)
                continue
            mod.register_forward_hook(self.inverter.get_layer_fwd_hook(mod))
            if isinstance(mod, torch.nn.ReLU):
                mod.register_full_backward_hook(self.relu_hook_function)

    @staticmethod
    def relu_hook_function(module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0),)

    def innvestigate(self, in_tensor):
        with torch.no_grad():
            self.inverter.module_list = []
            prediction = self.model(in_tensor)
            org_shape = prediction.size()
            prediction = prediction.view(org_shape[0], -1)
            max_v, _ = torch.max(prediction, dim=1, keepdim=True)
            only_max_score = torch.zeros_like(prediction)
            only_max_score[max_v == prediction] = prediction[max_v == prediction]
            relevance = only_max_score.view(org_shape)
            prediction.view(org_shape)

            rev_model = self.inverter.module_list[::-1]
            for layer in rev_model:
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
            return prediction, relevance


# import torchvision
# from torch.utils.data import DataLoader
# from torchvision.transforms import transforms
# from matplotlib import pyplot as plt
# import numpy as np
# from cnn_brain import ConvNet1
#
# nn_oasis = ConvNet1(num_classes=4)
# nn_oasis.load_state_dict(torch.load('../oasis_model.pth', map_location=torch.device('cpu')))
# # Convert to innvestigate model
# inn_model = InnvestigateModel(nn_oasis, lrp_exponent=2)
#
# train_path = '../TRAIN'
# test_path = '../TEST/test'
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# test_loader=DataLoader(
#     torchvision.datasets.ImageFolder(test_path,transform=transform),
#     batch_size=1, shuffle=True
# )
# for i, (data, target) in enumerate(test_loader):
#     if i > 0:
#         break
#     batch_size = int(data.size()[0])
#     model_prediction, true_relevance = inn_model.innvestigate(in_tensor=data)
#
#     vmin = np.percentile(true_relevance.reshape((-1,)), 35)
#     vmax = np.percentile(true_relevance.reshape((-1,)), 99)
#
#     plt.imshow(true_relevance[0].mean(axis=0), cmap='hot', vmin=vmin, vmax=vmax)
# plt.colorbar()
# plt.show()