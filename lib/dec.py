import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import math
from lib.utils import acc
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans


def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class DEC(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
        encodeLayer=[400], activation="relu", dropout=0, alpha=1.):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout) # f(x) = z
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim) # clustering layer -> q

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        # compute q -> NxK
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, q

    def encodeBatch(self, dataloader, islabel=False):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        encoded = []
        ylabels = []
        self.eval()
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = Variable(inputs)
            z,_ = self.forward(inputs)
            encoded.append(z.data.cpu())
            ylabels.append(labels)

        encoded = torch.cat(encoded, dim=0)
        ylabels = torch.cat(ylabels)
        if islabel:
            out = (encoded, ylabels)
        else:
            out = encoded
        return out

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p
    
    def predict(self, X, y):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        _, q = self.forward(X)

        # evalute the clustering performance
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        y = y.data.cpu().numpy()
        if y is not None:
            print("acc: %.5f, nmi: %.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred)))
            final_acc = acc(y, y_pred)
            final_nmi = normalized_mutual_info_score(y, y_pred)
        return final_acc, final_nmi

    def fit(self, X, y=None, lr=0.001, batch_size=256, num_epochs=10, update_interval=1, tol=1e-3):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Training DEC=======")
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.n_clusters, n_init=20)
        data, _ = self.forward(X)
        y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        y_pred_last = y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
            y = y.cpu().numpy()
            print("Kmeans acc: %.5f, nmi: %.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred)))

        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        print("num_batches:", num_batch)
        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                _, q = self.forward(X)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                if y is not None:
                    print("epoch: %.5f, acc: %.5f, nmi: %.5f" % (epoch, acc(y, y_pred), normalized_mutual_info_score(y, y_pred)))
                    final_acc = acc(y, y_pred)
                    final_nmi = normalized_mutual_info_score(y, y_pred)
                    final_epoch = epoch

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num
                y_pred_last = y_pred
                if epoch>0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    for i in range(self.n_clusters):
                        print("Cluster %d: %d" % (i, np.sum(y_pred==i)))
                    
                    break

                if epoch == num_epochs-1:
                    print("Reach max epoch. Stopping training.")
                    for i in range(self.n_clusters):
                        print("Cluster %d: %d" % (i, np.sum(y_pred==i)))


            # train 1 epoch
            train_loss = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                target = Variable(pbatch)

                z, qbatch = self.forward(inputs)
                loss = self.loss_function(target, qbatch)
                train_loss += loss.data*len(inputs)
                loss.backward()
                optimizer.step()

            #print("#Epoch %3d: Loss: %.4f" % (
            #    epoch+1, train_loss / num))

        return final_acc, final_nmi, final_epoch



