import os
import torch
import metrics as mt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from munkres import Munkres
import csv

class GraphConvSparse(nn.Module):
    """Create GraphConvSparse class, this is used to create the hidden layer, mean layer, and logstd layer

    Args:
        nn: the parameters to initialize the class, such as seed, number of featuers, neurons, etc.
    """
    def __init__(self, seed, input_dim, output_dim, activation = torch.sigmoid, **kwargs):
        """Initialize the GraphConvSparse class

        Args:
            seed: The seed used to control randomization
            input_dim: Input dimension
            output_dim: Output dimension
            activation (_type_, optional): Activation function for the graphs. Defaults to torch.sigmoid.
        """
        super(GraphConvSparse, self).__init__(**kwargs)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.weight = random_uniform_init(input_dim, output_dim, seed) 
        self.activation = activation
        
    def forward(self, inputs, adj):
        """Apllies the layer to the input objects

        Args:
            inputs: Hidden layer of feature matrix
            adj: Adjacency matrix

        Returns:
            _type_: Return the output from the activation function
        """
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs

class GMM_VGAE(nn.Module):
    """A Gaussian Mixture Model based variational graph autoencoder

    Args:
        nn: Inputs for intialization 
    """
    def __init__(self, **kwargs):
        super(GMM_VGAE, self).__init__()
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters = kwargs['nClusters']
        if kwargs['activation'] == "ReLU":
            self.activation = torch.relu
        if kwargs['activation'] == "Sigmoid":
            self.activation = torch.sigmoid
        if kwargs['activation'] == "Tanh":
            self.activation = torch.tanh
        self.seed = kwargs['seed']
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # VGAE training parameters
        self.base_gcn = GraphConvSparse(self.seed, self.num_features, self.num_neurons, self.activation)
        self.gcn_mean = GraphConvSparse( self.seed,self.num_neurons, self.embedding_size, activation = lambda x:x)
        self.gcn_logstddev = GraphConvSparse( self.seed,self.num_neurons, self.embedding_size, activation = lambda x:x)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # GMM training parameters    
        self.pi = nn.Parameter(torch.ones(self.nClusters)/self.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size),requires_grad=True)
                                  
    def pretrain(self, adj, features, adj_label, y, weight_tensor, norm, optimizer, epochs, lr, save_path, dataset, features_new):
        """Pretrain the model, saves the model to model.pk

        Args:
            adj: Adjacency matrix
            features: Feature matrix
            adj_label: Adjacency Label
            y: Truth Label
            weight_tensor: Weight Tensor
            norm: Normalization
            optimizer: Selected optimzer
            epochs: Amount of Epoch
            lr: Learning Rate
            save_path: Save path
            dataset: Dataset name

        Returns:
            _type_: Accuracy list
        """
        if  not os.path.exists(save_path + dataset + '/pretrain/model.pk'):
            if optimizer == "Adam":
                opti = Adam(self.parameters(), lr=lr)
            elif optimizer == "SGD":
                opti = SGD(self.parameters(), lr=lr, momentum=0.9)
            elif optimizer == "RMSProp":
                opti = RMSprop(self.parameters(), lr=lr)
            print('Pretraining......')
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            print(torch.cuda.is_available())
            print(torch.cuda.current_device())
            print(torch.cuda.device(0))
            print(torch.cuda.device_count())
            print(torch.cuda.get_device_name(0))
            # initialisation encoder weights
            epoch_bar = tqdm(range(epochs))
            acc_best = 0
            sc_best = -1
            gmm = GaussianMixture(n_components = self.nClusters , covariance_type = 'diag')
            acc_list = []
            for _ in epoch_bar:
                opti.zero_grad()
                # Get Z from the Encoder
                _,_, z = self.encode(features, adj)
                # Get x from the docder
                x_ = self.decode(z)
                # Calculate loss
                loss = norm*F.binary_cross_entropy(x_.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
                loss.backward()
                opti.step()
                epoch_bar.write('Loss pretraining = {:.4f}'.format(loss))
                y_pred = gmm.fit_predict(z.detach().cpu().numpy())
                print("pred_gmm : ", y_pred)
                print("Pred unique labels : ", set(y_pred))
                print("Pred length : ", len(y_pred))
                self.pi.data = torch.from_numpy(gmm.weights_)
                self.mu_c.data = torch.from_numpy(gmm.means_)
                self.log_sigma2_c.data =  torch.log(torch.from_numpy(gmm.covariances_))

                acc = mt.acc(y, y_pred)
                acc_list.append(acc)
                if (acc > acc_best):
                  acc_best = acc
                  self.logstd = self.mean 
                  torch.save(self.state_dict(), save_path + dataset + '/pretrain/model.pk')

                # sc = silhouette_score(features_new, y_pred, metric="cosine")
                # if (sc > sc_best):
                #   sc_best = sc
                #   self.logstd = self.mean 
                #   torch.save(self.state_dict(), save_path + dataset + '/pretrain/model.pk')

            print("Best accuracy : ",acc_best)
            return acc_list
        else:
            self.load_state_dict(torch.load(save_path + dataset + '/pretrain/model.pk'))
      
    def ELBO_Loss(self, features, adj, x_, adj_label, y, weight_tensor, norm, z_mu, z_sigma2_log, emb, L=1):
        """_summary_

        Args:
            features: Feature matrix
            adj: Adjacency matrix
            x_: x_ from decoding
            adj_label: Adjacency label
            y: Truth Label
            weight_tensor: _description_
            norm: _description_
            z_mu: Mean layer
            z_sigma2_log: logstd layer
            emb: Z from encoding
            L (int, optional): L. Defaults to 1.

        Returns:
            _type_: Loss ELBO, Loss reconstruction, Loss Clustering
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        pi = self.pi
        mu_c = self.mu_c
        log_sigma2_c = self.log_sigma2_c
        det = 1e-2 
        Loss_recons = 1e-2 * norm * F.binary_cross_entropy(x_.view(-1), adj_label, weight = weight_tensor)
        Loss_recons = Loss_recons * features.size(0)
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(emb, mu_c,log_sigma2_c)) + det
        yita_c = yita_c / (yita_c.sum(1).view(-1,1))
        y_pred = self.predict(emb)
        
        KL1 = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(0)) +
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2) / torch.exp(log_sigma2_c.unsqueeze(0)),2),1))
        KL2 = torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * torch.mean(torch.sum(1 + z_sigma2_log, 1))
        Loss_clus = KL1 - KL2
        
        Loss_elbo =  Loss_recons + Loss_clus 
        return Loss_elbo, Loss_recons, Loss_clus 
   
    def train(self, acc_list, adj_norm, features, adj_label, y, weight_tensor, norm, optimizer, epochs, lr, save_path, dataset, features_new):
        """Training the model

        Args:
            acc_list: List to store acc
            adj_norm: Processed graph
            features: Feature matrix
            adj_label: Adjacency Label
            y: Truth Label
            weight_tensor: Weight Tensor
            norm: Normalization
            optimizer: Selected optimzer
            epochs: Amount of Epoch
            lr: Learning Rate
            save_path: Save path
            dataset: Dataset name


        Returns:
            _type_: Final list of interested parameters, y prediction
        """
        self.load_state_dict(torch.load(save_path + dataset + '/pretrain/model.pk'))
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if optimizer ==  "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay = 0.01)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay = 0.01)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay = 0.01)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)
        
        import csv, os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Check if the device is cuda available, error if not
        print(torch.cuda.is_available())
        print(torch.cuda.current_device())
        print(torch.cuda.device(0))
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))

        # Logging the resluts
        logfile = open(save_path + dataset + '/cluster/log.csv', 'w')
        # logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'f1_macro', 'f1_micro', 'precision_macro', 'precision_micro', 'Loss_recons', 'Loss_clus' , 'Loss_elbo'])
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'silhouette','davies', 'calinski', 'Loss_elbo'])
        logwriter.writeheader()
        
        epoch_bar=tqdm(range(epochs))
        
        print('Training......')
        
        count =0
        currmax = 0
        finalist = []
        for epoch in epoch_bar:
            opti.zero_grad()
            # Encoding
            z_mu, z_sigma2_log, emb = self.encode(features, adj_norm) 
            # Decoding
            x_ = self.decode(emb)
            # Loss calculation 
            Loss_elbo, Loss_recons, Loss_clus = self.ELBO_Loss(features, adj_norm, x_, adj_label.to_dense().view(-1), y, weight_tensor, norm, z_mu , z_sigma2_log, emb)
            epoch_bar.write('Loss={:.4f}'.format(Loss_elbo.detach().cpu().numpy()))

            # Prediction and metrics
            y_pred = self.predict(emb)
            cm = clustering_metrics(y, y_pred)
            acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro = cm.evaluationClusterModelFromLabel()
            acc_list.append(acc)
            sc = silhouette_score(features_new, y_pred, metric="cosine")
            calinski = calinski_harabasz_score(features_new, y_pred)
            davies = davies_bouldin_score(features_new, y_pred)
            
            #Save logs 
            # logdict = dict(iter = epoch, acc = acc, nmi= nmi, ari=adjscore, f1_macro=f1_macro , f1_micro=f1_micro, precision_macro=precision_macro, precision_micro = precision_micro, Loss_recons=Loss_recons.detach().cpu().numpy(), Loss_clus=Loss_clus.detach().cpu().numpy(), Loss_elbo=Loss_elbo.detach().cpu().numpy())
            logdict = dict(iter = epoch, acc = acc, nmi= nmi, ari=adjscore, silhouette= sc, calinski=calinski, davies=davies, Loss_elbo=Loss_elbo.detach().cpu().numpy())
            logwriter.writerow(logdict)
            logfile.flush() 
            
            Loss_elbo.backward()
            opti.step()
            lr_s.step()
            count+=1
            if adjscore>currmax:
                finalist = [acc, adjscore, Loss_recons.detach().cpu().numpy(),Loss_clus.detach().cpu().numpy(),Loss_elbo.detach().cpu().numpy(), epoch]
                currmax = adjscore
        return finalist, y_pred, y
               
    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        """Calculate gaussian pdfs log

        Args:
            x: Z from encoding
            mus: Mean layer
            log_sigma2s: logstd layer

        Returns:
            _type_: Calculated result
        """
        G=[]
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    def gaussian_pdf_log(self,x,mu,log_sigma2):
        """Calculate gaussian pdf log

        Args:
            x: Z from encoding
            mus: Mean layer
            log_sigma2s: logstd layer

        Returns:
            _type_: Calculated result
        """
        c = -0.5 * torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1)
        return c

    def predict(self, z):
        """Predict the cluster label

        Args:
            z: Z matrix from encoder

        Returns:
            _type_: Prediction in list
        """
        pi = self.pi
        log_sigma2_c = self.log_sigma2_c  
        mu_c = self.mu_c
        det = 1e-2
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)

    def encode(self, x_features, adj):
        """Encoder of GMM-VGAE

        Args:
            x_features: Feature matrix
            adj: Adjacency matrix

        Returns:
            _type_: Mean layer, logstd layer, z matrix
        """
        hidden = self.base_gcn(x_features, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        gaussian_noise = torch.randn(x_features.size(0), self.embedding_size)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return self.mean, self.logstd, sampled_z
            
    @staticmethod
    def decode(z):
        """Docder

        Args:
            z: Z matrix from the encoder

        Returns:
            _type_: Reconstructed graph
        """
        A_pred = torch.sigmoid(torch.matmul(z,z.t()))
        return A_pred
        
def random_uniform_init(input_dim, output_dim, seed):
    """Create Gaussian random noise

    Args:
        input_dim: Input dimension
        output_dim: Output diemsnion
        seed: Seed

    Returns:
        _type_: Random noise
    """
    np.random.seed(seed)
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    torch.manual_seed(seed)
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)
  
class clustering_metrics():
    """Class for calculation of metrics
    """
    def __init__(self, true_label, predict_label):
        """Initialize the clustering metrics class

        Args:
            true_label: The true label
            predict_label: The predicted label
        """
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        """Calculate the best mapping between true and predict label

        Returns:
            _type_: Accuracy 
        """
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.true_label))
        numclass2 = len(l2)

        if numclass1 != numclass2:
            print(numclass1)
            print(numclass2)
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]
            

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c
        acc = metrics.accuracy_score(self.true_label, new_predict)
        
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro', zero_division=1)
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro', zero_division=1)
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        """Calculate evaluation metrics using sklearn functions

        Returns:
            _type_: List of interested metrics
        """
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

        fh = open('recoder.txt', 'a')

        fh.write('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore) )
        fh.write('\r\n')
        fh.flush()
        fh.close()

        return acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro