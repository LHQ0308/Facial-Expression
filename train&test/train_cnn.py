from layer.core import *
from algorithm.SGD import Mini_Batch
from data.process import loadTrainData
from layer.model import Model
if __name__ == '__main__':
    # load ROI+ROTATION dataset
    dataSet=loadTrainData("dataset/ROT/data.pkl","dataset/ROT/mean.pkl");
    # load ROI dataset
    #dataSet=loadTrainData("dataset/ROI/data.pkl","dataset/ROI/mean.pkl");
    # load normal dataset
    #dataSet=loadTrainData("dataset/NORMAL/data.pkl","dataset/NORMAL/mean.pkl");
    cifar=Model(batch_size=100,lr=0.0001,dataSet=dataSet,weight_decay=0)
    neure=[64,64,128,300]
    #neure=[32,32,64,64]
    #neure=[48,48,96,200]
    batch_size=100
    cifar.add(DataLayer(batch_size,(32,32,1)))
    cifar.add(ConvolutionLayer((batch_size,1,32,32),(neure[0],1,3,3),'relu','Gaussian',0.0001))
    cifar.add(PoolingLayer())
    cifar.add(ConvolutionLayer((batch_size,neure[0],15,15),(neure[1],neure[0],4,4),'relu','Gaussian',0.01))
    cifar.add(PoolingLayer())
    cifar.add(ConvolutionLayer((batch_size,neure[1],6,6),(neure[2],neure[1],5,5),'relu','Gaussian',0.01))
    cifar.add(PoolingLayer())
    cifar.add(FullyConnectedLayer(neure[2]*1*1,neure[3],'relu','Gaussian',0.1))
    cifar.add(DropoutLayer(0.5))
    cifar.add(SoftmaxLayer(neure[3],5,'Gaussian',0.1))
    cifar.build_train_fn()
    cifar.build_vaild_fn()
    algorithm=Mini_Batch(model=cifar,n_epochs=100,load_param='params/CNN128_ROT.pkl',save_param='cnn_params.pkl')
    algorithm.run()
    