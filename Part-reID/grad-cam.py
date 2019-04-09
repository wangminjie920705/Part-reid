import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
from network5 import MGN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='/home/wangminjie/Desktop/wmj/projects/Part-reID_2/CAM/1.png',help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        x=model(x)
        x.register_hook(self.save_gradient)
        outputs += [x]
        return outputs, x


class ModelOutputs():
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self,x):
        target_activations, output = self.feature_extractor(x)
        output=model.avgpool3(output)
        output=output.view(output.size(0),-1)
        output=model.layer3_fc(output)
        return target_activations,output



class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self,input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()  ##这两行同理，features不包含，可以重新加回去试一试，会报错不包含这个对象。
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)  ##这里适配我们的torch0.4及以上，我用的1.0也可以完美兼容。（variable改成graph即可）

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (128, 384))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("./CAM/layer3.jpg", np.uint8(255 * cam))


if __name__ == '__main__':
    args = get_args()
    model = MGN()
    model.load_state_dict\
        (torch.load('/home/wangminjie/Desktop/wmj/projects/Part-reID_2/weights/CA+PA_labeled/model_800.pt'))
    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (128, 384))) / 255
    input = preprocess_image(img)
    print('input.size()=', input.size())

    grad_cam = GradCam(model,target_layer_names=["layer4"], use_cuda=args.use_cuda)

    target_index = None
    mask = grad_cam(input, target_index)
    show_cam_on_image(img, mask)




