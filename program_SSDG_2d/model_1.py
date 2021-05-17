import torch.nn as nn
import torch
from torch.autograd import Function
import resnet

def l2_norm(input, axis=1):
    norm = torch.norm(input, p = 2, dim = axis, keepdim = True)
    output = torch.div(input, norm)
    return output

class Extractor(nn.Module):
    def __init__(self, args):
        super(Extractor, self).__init__()
        self.args = args
        self.resnet_ = resnet.resnet18(pretrained = True, progress = True)
        self.hallucinate = HallucinationNet(args)
        self.resnet_.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
        )

    def augment(self, features):
        fake_features = []
        batch_size, channels = features.shape
        for select_index in range(batch_size):
            noise = torch.randn(self.args.h_degree, 512).cuda()
            # feauture_noise = torch.cat([features[select_index].unsqueeze(0).expand(self.args.h_degree, -1), noise], dim = 1)
            feauture_noise = features[select_index].unsqueeze(0).expand(self.args.h_degree, -1) + noise
            feauture_noise = nn.ReLU(True)(feauture_noise)
            fake_features.append(self.hallucinate(feauture_noise).unsqueeze(0))
        features = features.unsqueeze(1)
        noise_features = torch.cat(fake_features, dim = 0)
        mix_feature = torch.mean(torch.cat([features, noise_features], dim = 1), dim = 1).squeeze(1)
        return mix_feature, noise_features

    def forward(self, input):
        features = self.resnet_(input)  #(b,512)
        batch_size, feature_len = features.shape
        mix_features, noise_features = self.augment(features)
        mix_features_norm = (torch.norm(mix_features, dim = 1, keepdim = True) * 2) ** 0.5
        mix_features = torch.div(mix_features, mix_features_norm)
        noise_features_norm = (torch.norm(noise_features, dim = 2, keepdim = True) * 2) ** 0.5
        noise_features = torch.div(noise_features, noise_features_norm)
        features = features.reshape(batch_size, -1, feature_len).expand(-1, self.args.h_degree, -1)
        features_norm = (torch.norm(features, dim = 2, keepdim = True) * 2) ** 0.5
        features = torch.div(features, features_norm)
        return mix_features, noise_features, features

class HallucinationNet(nn.Module):
    def __init__(self, args, in_channels = 512, hidden = 512, out_channels = 512):
        super(HallucinationNet, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
                            nn.Linear(in_channels, out_channels),
                            nn.ReLU(True),
                       )
        self.encoder[0].weight.data.copy_(torch.eye(in_channels))
        # self.encoder[2].weight.data.copy_(torch.eye(in_channels))
        # self.__initialize_weights()

    def forward(self, feauture_noise):
        fake_feature = self.encoder(feauture_noise)
        return fake_feature

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

class Classifier(nn.Module):
    def __init__(self, in_channels = 512, hidden = 512, out_channels = 2):
        super(Classifier, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(in_channels, out_channels),
        )
        # self.classify = nn.Sequential(
            # nn.Linear(in_channels, 2 * in_channels),
            # nn.ReLU(True),
            # nn.Linear(2 * in_channels, hidden),
            # nn.ReLU(True),
            # nn.Linear(hidden, out_channels),
        # )
        self.__initialize_weights()

    def forward(self, input):
        for module in self.classify:
            try:
                module.weight.data = l2_norm(module.weight, axis = 0)
            except:
                pass
        result = self.classify(input)
        return result

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

class Discriminator(nn.Module):
    def __init__(self, in_channels = 512, hidden = 512, out_channels = 2):
        super(Discriminator, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )
        # self.classify = nn.Sequential(
        #     nn.Linear(in_channels, 2 * in_channels),
        #     nn.ReLU(True),
        #     nn.Linear(2 * in_channels, hidden),
        #     nn.ReLU(True),
        #     nn.Linear(hidden, out_channels),
        # )
        self.__initialize_weights()

    def forward(self, input, lambda_term):
        reverse_input = Gradient_Reverse_Layer.apply(input, lambda_term)
        result = self.classify(reverse_input)
        return result

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

class Gradient_Reverse_Layer(Function):
    @staticmethod
    def forward(ctx, input, lambda_term):
        ctx.lambda_term = lambda_term
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_term
        return output, None