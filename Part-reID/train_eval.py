import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch.optim import lr_scheduler

from opt import opt
from data import Data
from network5 import MGN
from loss import Loss
from utils import get_optimizer,extract_feature
from metrics import mean_ap, cmc, re_ranking
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import visdom

class Main():
    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.to('cuda')
        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

        self.trackloss=0

        # self.track_loss = 0
        # self.global_step = 0
        # self.vis = visdom.Visdom(env=u"train_loss")
        # self.win = self.vis.line(X=np.array([self.global_step]), Y=np.array([self.track_loss]))

    def train(self):

        self.scheduler.step()

        self.model.train()
        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            self.trackloss=loss
            # if self.global_step%100==0:
            #     self.vis.line(X=np.array([self.global_step]), Y=np.array([loss.data[0]]), win=self.win,update='append')
            # self.global_step += 1
            loss.backward()
            self.optimizer.step()

    def evaluate(self):

        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        r, m_ap = rank(dist)

        print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))


        #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

    def vis(self):

        self.model.eval()

        gallery_path = data.testset.imgs
        query_path=data.queryset.imgs

        # Extract feature
        print('extract features, this may take a few minutes')
        #query_feature = extract_feature(model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
        query_feature=extract_feature(model, tqdm(data.query_loader))
        gallery_feature = extract_feature(model, tqdm(data.test_loader))
        print(query_feature.size())

        for query_Index in range(query_feature.size()[0]):
            # sort images
            query_img_path=query_path[query_Index]
            query_feature_now = query_feature[query_Index].unsqueeze(1)
            print(query_feature_now.size())
            #query_feature = query_feature.view(-1, 1)
            #print(query_feature.size())
            score = torch.mm(gallery_feature, query_feature_now)
            score = score.squeeze(1).cpu()
            score = score.numpy()

            index = np.argsort(score)  # from small to large
            index = index[::-1]  # from large to small

            # # Remove junk images
            # junk_index = np.argwhere(gallery_label == -1)
            # mask = np.in1d(index, junk_index, invert=True)
            # index = index[mask]

            # Visualize the rank result
            fig = plt.figure(figsize=(16, 4))

            ax = plt.subplot(1, 11, 1)
            ax.axis('off')
            plt.imshow(plt.imread(query_img_path))
            ax.set_title('query')

            #print('Top 10 images are as follow:')

            for i in range(10):
                img_path = gallery_path[index[i]]
                #print(img_path)

                ax = plt.subplot(1, 11, i + 2)
                ax.axis('off')
                plt.imshow(plt.imread(img_path))
                ax.set_title(img_path.split('/')[-1][:9])

            fig.savefig("/home/wangminjie/Desktop/wmj/projects/Part-reID_2/result/detect/{}.png".format(query_img_path.split('/')[-1][:9]))
            print('result saved to show{}.png'.format(query_img_path.split('/')[-1][:9]))
        print("task end")




if __name__ == '__main__':

    data = Data()
    model = MGN()
    loss = Loss()
    reid = Main(model, loss, data)

    # track_loss = 0
    # global_step = 0
    # vis = visdom.Visdom(env=u"train_loss")
    # win = vis.line(X=np.array([global_step]), Y=np.array([track_loss]))

    if opt.mode == 'train':

        for epoch in range(1, opt.epoch+1):
            print('\nepoch', epoch)
            reid.train()
            # vis.line(X=np.array([global_step]), Y=np.array([reid.trackloss.data[0]]), win=win,update='append')
            # global_step += 1
            if epoch % 100 == 0:
                print('\nstart evaluate')
                reid.evaluate()
                os.makedirs('weights',exist_ok=True)
                torch.save(model.state_dict(), ('weights/model_{}.pt'.format(epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        reid.evaluate()

    if opt.mode == 'vis':
        print('visualize')
        model.load_state_dict(torch.load(opt.weight))
        reid.vis()
