from PIL import Image
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.models as models


class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retrain_graph=retain_graph)
        return self.loss


def load_image(path):
    image = Image.open(path)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image


def show_image(tensor, title=None):
    image = tensor.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


imsize = 512 if use_cuda else 128

loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

style_img = load_image("").type(dtype)
content_img = load_image("").type(dtype)

assert style_img.size() == content_img.size(), \
    "Style and content image must have the same size"


unloader = transforms.ToPILImage()

plt.ion()

plt.figure()
show_image(style_img.data, title='Style Image')

plt.figure()
show_image(content_img.data, title='Content Image')

cnn = models.vgg19(pretrained=True).features

if use_cuda:
    cnn = cnn.cuda()

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, style_img, content_img, style_weight=1000, content_weight=1,
                               content_layers=content_layers_default, style_layers=style_layers_default):

    cnn = copy.deepcopy(cnn)

    content_losses = []
    style_losses = []

    model = nn.Sequential()
    gram = GramMatrix()

    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)











