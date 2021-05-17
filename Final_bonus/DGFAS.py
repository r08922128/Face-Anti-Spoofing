import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import sys
import numpy as np
from torch.autograd import Variable
import random
import os
from torch.autograd import Function

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Feature_Generator_MADDG(nn.Module):
    def __init__(self):
        super(Feature_Generator_MADDG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(128)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv1_3 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(196)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.conv1_4 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(128)
        self.relu1_4 = nn.ReLU(inplace=True)
        self.maxpool1_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_5 = nn.BatchNorm2d(128)
        self.relu1_5 = nn.ReLU(inplace=True)
        self.conv1_6 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_6 =  nn.BatchNorm2d(196)
        self.relu1_6 = nn.ReLU(inplace=True)
        self.conv1_7 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_7 = nn.BatchNorm2d(128)
        self.relu1_7 = nn.ReLU(inplace=True)
        self.maxpool1_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_8 = nn.BatchNorm2d(128)
        self.relu1_8 = nn.ReLU(inplace=True)
        self.conv1_9 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_9 = nn.BatchNorm2d(196)
        self.relu1_9 = nn.ReLU(inplace=True)
        self.conv1_10 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_10 = nn.BatchNorm2d(128)
        self.relu1_10 = nn.ReLU(inplace=True)
        self.maxpool1_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.bn1_1(out)
        out = self.relu1_1(out)
        out = self.conv1_2(out)
        out = self.bn1_2(out)
        out = self.relu1_2(out)
        out = self.conv1_3(out)
        out = self.bn1_3(out)
        out = self.relu1_3(out)
        out = self.conv1_4(out)
        out = self.bn1_4(out)
        out = self.relu1_4(out)
        pool_out1 = self.maxpool1_1(out)

        out = self.conv1_5(pool_out1)
        out = self.bn1_5(out)
        out = self.relu1_5(out)
        out = self.conv1_6(out)
        out = self.bn1_6(out)
        out = self.relu1_6(out)
        out = self.conv1_7(out)
        out = self.bn1_7(out)
        out = self.relu1_7(out)
        pool_out2 = self.maxpool1_2(out)

        out = self.conv1_8(pool_out2)
        out = self.bn1_8(out)
        out = self.relu1_8(out)
        out = self.conv1_9(out)
        out = self.bn1_9(out)
        out = self.relu1_9(out)
        out = self.conv1_10(out)
        out = self.bn1_10(out)
        out = self.relu1_10(out)
        pool_out3 = self.maxpool1_3(out)
        return pool_out3

class Feature_Embedder_MADDG(nn.Module):
    def __init__(self):
        super(Feature_Embedder_MADDG, self).__init__()
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottleneck_layer_1 = nn.Sequential(
            self.conv3_1,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.pool2_1,
            self.conv3_2,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self.pool2_2,
            self.conv3_3,
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self, input, norm_flag):
        feature = self.bottleneck_layer_1(input)
        feature = self.avg_pool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)
        if (norm_flag):
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
        return feature

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # change your path
    model_path = './resnet18-5c106cde.pth'
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("loading model: ", model_path)
    # print(model)
    return model

class Feature_Generator_ResNet18(nn.Module):
    def __init__(self):
        super(Feature_Generator_ResNet18, self).__init__()
        model_resnet = resnet18(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        return feature

class Feature_Embedder_ResNet18(nn.Module):
    def __init__(self):
        super(Feature_Embedder_ResNet18, self).__init__()
        model_resnet = resnet18(pretrained=False)
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input, norm_flag):
        feature = self.layer4(input)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)
        # if (norm_flag):
        #     feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
        #     feature = torch.div(feature, feature_norm)
        
        return feature


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 3)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag = True):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out

# class GRL(torch.autograd.Function):
#     # def __init__(self):
#     #     self.iter_num = 0
#     #     self.alpha = 10
#     #     self.low = 0.0
#     #     self.high = 1.0
#     #     self.max_iter = 4000  # be same to the max_iter of config.py
#     @staticmethod
#     def forward(ctx, input,iter_num):
#         ctx.iter_num = iter_num+1
#         return input * 1.0
#     @staticmethod
#     def backward(ctx, gradOutput):
#         coeff = np.float(2.0 * (1.0 - 0.0) / (1.0 + np.exp(-10 * ctx.iter_num / 4000))
#                          - (1.0 - 0.0) + 0.0)
#         return -coeff * gradOutput
class Gradient_Reverse_Layer(Function):
    @staticmethod
    def forward(ctx, input, lambda_term):
        ctx.lambda_term = lambda_term
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_term
        return output, None

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(512, 2)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        #self.grl_layer = GRL()
    #@staticmethod
    def forward(self, feature,lambda_term):
        reverse_input = Gradient_Reverse_Layer.apply(feature, lambda_term)
        adversarial_out = self.ad_net(reverse_input)
        #adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out

class HallucinationNet(nn.Module):
    def __init__(self, args, in_channels = 1024, hidden = 512, out_channels = 512):
        super(HallucinationNet, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
                            nn.Linear(in_channels, hidden),
                            nn.ReLU(True),
                            nn.Linear(hidden, out_channels),
                            nn.ReLU(True),
                       )
        # self.encoder[0].weight.data.copy_(torch.eye(in_channels))
        # self.encoder[2].weight.data.copy_(torch.eye(hidden))
        # self.__initialize_weights()

    def forward(self, feauture_noise):
        fake_feature = self.encoder(feauture_noise)
        return fake_feature

class DG_model(nn.Module):
    def __init__(self, model,args):
        super(DG_model, self).__init__()
        if(model == 'resnet18'):
            self.backbone = Feature_Generator_ResNet18()
            self.embedder = Feature_Embedder_ResNet18()
        elif(model == 'maddg'):
            self.backbone = Feature_Generator_MADDG()
            self.embedder = Feature_Embedder_MADDG()
        else:
            print('Wrong Name!')
        self.classifier = Classifier()
        self.args = args
        self.hallucinate = HallucinationNet(args)
    def augment(self, features):
        fake_features = []
        batch_size, channels = features.shape
        for select_index in range(batch_size):
            noise = torch.randn(self.args.h_degree, 512).cuda()
            feauture_noise = torch.cat([features[select_index].unsqueeze(0).expand(self.args.h_degree, -1), noise], dim = 1)
            # feauture_noise = features[select_index].unsqueeze(0).expand(self.args.h_degree, -1) + noise
            fake_features.append((self.hallucinate(feauture_noise)).unsqueeze(0))
        noise_features = torch.cat(fake_features, dim = 0)
        noise_features = torch.cat([features.unsqueeze(1), noise_features], dim = 1)
        # noise_features = torch.mean(torch.cat([features.unsqueeze(1), noise_features], dim = 1), dim = 1).squeeze(1)
        return noise_features
        

    def forward(self, input, norm_flag = True):
        #print(input.size())#torch.Size([32, 3, 512, 512])
        features = self.backbone(input)
        #print(feature.size())#torch.Size([32, 256, 32, 32])
        clean_features = self.embedder(features, norm_flag)

        ##clean feature
        if (norm_flag):
            clean_features_no_norm = clean_features
            feature_norm = torch.norm(clean_features, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            clean_features = torch.div(clean_features, feature_norm)
        clean_classifier_out=self.classifier(clean_features, norm_flag)

        ##aug feature
        batch_size, feature_len = clean_features.shape
        aug_features = self.augment(clean_features)

        # noise_features_norm = (torch.norm(noise_features, dim = 2, keepdim = True) * 2) ** 0.5
        # noise_features = torch.div(noise_features, noise_features_norm)
        #print(aug_features.size())#torch.Size([32, 10, 512])
        # features=features.unsqueeze(1)
        # aug_features=torch.cat((features,noise_features),dim=1)
        aug_features=aug_features.view(-1,512)
        #print(aug_features.size())#torch.Size([32, 11, 512])
        if (norm_flag):
            aug_features_no_norm = aug_features
            feature_norm = torch.norm(aug_features, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            aug_features = torch.div(aug_features, feature_norm)
        aug_classifier_out = self.classifier(aug_features, norm_flag)
        return aug_classifier_out, aug_features, aug_features_no_norm, clean_classifier_out, clean_features, clean_features_no_norm

if __name__ == '__main__': 
    x = Variable(torch.ones(1, 3, 256, 256))
    model = DG_model()
    y, v = model(x, True)