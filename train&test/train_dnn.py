from layer.core import *
from algorithm.SGD import Mini_Batch
from data.process import loadTrainData
from layer.model import Model
if __name__ == '__main__':
    # load ROI+ROTATION dataset
    #dataSet=loadTrainData("dataset/ROT/data.pkl","dataset/ROT/mean.pkl",scale=128.0);
    # load ROI dataset
    dataSet=loadTrainData("dataset/ROI/data.pkl","dataset/ROI/mean.pkl",scale=128.0);
    # load normal dataset
    #dataSet=loadTrainData("dataset/NORMAL/data.pkl","dataset/NORMAL/mean.pkl",scale=128.0); 
    cifar=Model(batch_size=100,lr=0.001,dataSet=dataSet,weight_decay=0.0)
    #neure=[2000]
    neure=[1000,1000,1000]
    #neure=[2000,2000,2000];
    batch_size=100
    cifar.add(DataLayer(batch_size,32*32))
    cifar.add(FullyConnectedLayer(32*32,neure[0],'relu','Gaussian',0.1))
    cifar.add(DropoutLayer(0.2))
    cifar.add(FullyConnectedLayer(neure[0],neure[1],'relu','Gaussian',0.1))
    cifar.add(DropoutLayer(0.2))
    cifar.add(FullyConnectedLayer(neure[1],neure[2],'relu','Gaussian',0.1))
    cifar.add(DropoutLayer(0.2))     
    cifar.add(SoftmaxLayer(neure[2],5))
    cifar.build_train_fn()
    cifar.build_vaild_fn()
    algorithm=Mini_Batch(model=cifar,n_epochs=200,load_param='mlp_params.pkl',save_param='mlp_params.pkl')
    algorithm.run()