print("")

print("---------------")
print("Start System")
print("---------------")
#from matplotlib import pyplot as plt
#from torchvision import datasets
import os
import time

import lhcnn

print( "lhcnn loaded")

from CIFAR2 import CIFAR2



#=======================
#  数据载入
#=======================
print("---------------")
print("Load Dataset")
def load_dataset():
    global cifar2_train, cifar2_test
    
    transform = lhcnn.Compose([
        lhcnn.ToArray(), 
        lhcnn.Normalize( 128,  128 ), 
        lhcnn.Flatten(), 
    ])

    data_train = CIFAR2(train = True,  transform = transform)
    data_test = CIFAR2(train = False,  transform = transform)

    cifar2_train = lhcnn.DataLoader(data_train, batch_size=64, shuffle=True)
    cifar2_test = lhcnn.DataLoader(data_test, batch_size=64, shuffle=True)

load_dataset()

#=======================
#  training_loop
#=======================
def training_loop( n_epochs,  optimizer,  model,  loss_fn,  train_loader, prev_loops=0):
    
    for epoch in range(n_epochs):
        
        loss_train = 0.0
        st = time.time()
        
        for imgs,  labels in train_loader:
            
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            
            model.cleargrads()
        
            loss.backward()
            optimizer.update()
            
            loss_train += loss.data
        
        dt = time.time() - st
        print( "epoch %d, loss: %f,cost %3f sec"%(epoch+prev_loops,  float(loss_train), dt ) )

#=======================
#  创建模型
#=======================
print("---------------")
print("Create NN Model")

linear1 = lhcnn.Linear(3072, 512)
tanh1 = lhcnn.TanhFn()
linear2 = lhcnn.Linear(512, 2)
#softmax = lhcnn.LogSoftmaxFn(dim=1)

model = lhcnn.Sequential( linear1, tanh1, linear2 )


optimizer = lhcnn.SGD( lr=1e-3)
optimizer.setup(model)
    
loss_fn = lhcnn.SoftmaxCrossEntropy()
training_loop(100, optimizer, model, loss_fn, cifar2_train)

#numel_list = [p.numel() for p in model.parameters() if p.requires_grad == True]
#print( "numel_list:", numel_list)


