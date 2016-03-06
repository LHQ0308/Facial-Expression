from layer.core import *
from algorithm.SGD import Mini_Batch
from data.process import loadTestData
from layer.model import Model
from SocketServer import ThreadingTCPServer, StreamRequestHandler
import traceback
from collections import OrderedDict
test_hit=0
test_count=0
def test_single(filename,mean_filename,y,scale=128.0):
    size,dataSet=loadTestData(filename,mean_filename,scale)
    test_fn=theano.function(inputs=[index],outputs=[test_pred,test_belief],givens={Test_Single.x:dataSet[index*batch_size:(index+1)*batch_size]})
    ans=0
    for i in xrange(size):
        pred,belief=test_fn(i)
        if(pred[0]==y):ans=ans+1
    print "facial expression %d: test result %d/%d" %(y,ans,size)
    print "                     error rate %f%%" %(float(size-ans)/size*100)
    global test_hit,test_count;
    test_hit+=ans
    test_count+=size
def test_single_stat():
    print "\nNow tesing without KNN.....\n"
    global test_hit,test_count;
    for idx in xrange(5):
        test_single(("dataset/NORMAL_TEST/%d.bin")%idx,"dataset/ROI/mean.pkl",idx)
    print "total error rate: %f%%\n"%(float(test_count-test_hit)/test_count*100)
    #  clear
    test_hit=test_count=0
def test_knn(filename,mean_filename,y,scale=128.0):
    size,dataSet=loadTestData(filename,mean_filename,scale)
    vaild_fn=theano.function(inputs=[index],outputs=[test_pred,test_belief],givens={Test_KNN.x:dataSet[index*batch_size:(index+1)*batch_size]})
    examples=size/9
    ans=0
    stat=OrderedDict()
    for j in xrange(5): stat[j]=0
    for i in xrange(examples):
        pred,belief=vaild_fn(i)
        voter=OrderedDict()
        for j in xrange(5): voter[j]=0
        for j in pred:voter[j]=voter[j]+1
        tt=[]
        for j in xrange(5):tt.append(voter[j])
        tx=numpy.asarray(tt)
        #  increasing argsort
        idx=numpy.argsort(-tx)
        for j in xrange(1):
            if(tx[idx[j]]!=tx[idx[j+1]]):
                if(idx[j]==y):ans=ans+1
            # conservative predication
            else:
                if(y==0):ans=ans+1
    print "facial expression %d: test result %d/%d" %(y,ans,examples)
    print "                     error rate %f%%" %(float(examples-ans)/examples*100)
    global test_hit,test_count;
    test_hit+=ans
    test_count+=examples
def test_knn_stat():
    print "\nNow tesing with KNN-ROI.....\n"
    global test_hit,test_count;
    for idx in xrange(5):
        test_knn(("dataset/ROI_TEST/%d.bin")%idx,"dataset/ROI/mean.pkl",idx)
    print "total error rate: %f%%\n"%(float(test_count-test_hit)/test_count*100)
    #  clear
    test_hit=test_count=0    
    
if __name__ == '__main__':
    Test_Single=Model(batch_size=1,lr=0.01,dataSet=None)
    meta_num=100
    neure=[meta_num,meta_num,meta_num,meta_num]
    batch_size=1
    x=T.matrix('x')
    index=T.lscalar()
    Test_Single.add(DataLayer(batch_size,32*32))
    Test_Single.add(FullyConnectedLayer(32*32,neure[0],'relu','Gaussian',0.1))
    Test_Single.add(DropoutLayer(0.2))
    Test_Single.add(FullyConnectedLayer(neure[0],neure[1],'relu','Gaussian',0.1))
    Test_Single.add(DropoutLayer(0.2))
    Test_Single.add(FullyConnectedLayer(neure[1],neure[2],'relu','Gaussian',0.1))
    Test_Single.add(DropoutLayer(0.2))     
    Test_Single.add(SoftmaxLayer(neure[2],5))
    Test_Single.build_test_fn()
    Test_Single.load_params('params/DNN2000_ROI.pkl')
    test_pred=Test_Single.test_pred
    test_belief=Test_Single.test_belief
    test_single_stat()
    
    batch_size=9
    x=T.matrix('x')
    Test_KNN=Model(batch_size=9,lr=0.01,dataSet=None)
    Test_KNN.add(DataLayer(batch_size,(32,32,1)))
    Test_KNN.add(FullyConnectedLayer(32*32,neure[0],'relu','Gaussian',0.1))
    Test_KNN.add(DropoutLayer(0.2))
    Test_KNN.add(FullyConnectedLayer(neure[0],neure[1],'relu','Gaussian',0.1))
    Test_KNN.add(DropoutLayer(0.2))
    Test_KNN.add(FullyConnectedLayer(neure[1],neure[2],'relu','Gaussian',0.1))
    Test_KNN.add(DropoutLayer(0.2)) 
    Test_KNN.add(SoftmaxLayer(neure[3],5,'Gaussian',0.1))
    Test_KNN.build_test_fn()
    Test_KNN.load_params('params/DNN2000_ROI.pkl')
    test_pred=Test_KNN.test_pred
    test_belief=Test_KNN.test_belief
    test_knn_stat()
    
    