print("")
print("---------------")
print("Start System")

import numpy as np

import lhcnn
import time
from CIFAR2 import CIFAR2

RUN_ON_GPU = False
WEIGHT_FILE = "test_c01_ep1200.pt"

device = None

class Net(lhcnn.Layer):
    def __init__(self):
        super().__init__()
        
        self.conv1 = lhcnn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, pad=1)
        self.conv2 = lhcnn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, pad=1)
        self.fc1 = lhcnn.Linear(512, 32)
        self.fc2 = lhcnn.Linear(32, 2)
        
        
    def forward(self,  x):
        x = self.conv1(x)
        x = lhcnn.TanhFn()(x)
        x = lhcnn.Pooling(2, 2)(x )
        x = self.conv2(x)
        x = lhcnn.TanhFn()(x)
        x = lhcnn.Pooling(2, 2)(x)
        
        x = lhcnn.call_reshape(x, (x.shape[0], -1))
        
        x = self.fc1(x)
        x = lhcnn.TanhFn()(x)
        x = self.fc2(x)

        return x


#=======================
#  数据载入
#=======================
def load_dataset():
    global cifar2_train, cifar2_test
    
    transform = lhcnn.Compose([
        lhcnn.ToArray(), 
        lhcnn.Normalize( 128,  128 ), 
        #lhcnn.Flatten(), 
    ])

    data_train = CIFAR2(train = True,  transform = transform)
    data_test = CIFAR2(train = False,  transform = transform)

    data_train = CIFAR2(train = True,  transform = transform)
    data_test = CIFAR2(train = False,  transform = transform)

    cifar2_train = lhcnn.DataLoader(data_train, batch_size=64, shuffle=True)
    cifar2_test = lhcnn.DataLoader(data_test, batch_size=64, shuffle=True)

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
            
def test_model(model, train_loader, test_loader,  verbose = False):
    
    for name, loader in [('train', train_loader), ('test', test_loader)]:
    
        if verbose:
            print("run with %s data"%name)
    
        correct = 0
        total = 0
        with lhcnn.using_config('enable_backprop', False):
            for imgs,  labels in loader:
               
                batch_size = imgs.shape[0]
                
                outputs = model(imgs)
                softmax = lhcnn.call_softmax(outputs.data)
                predicted = np.argmax(softmax.data, axis=1)
                batch_correct = int((predicted == labels).sum())
                if verbose:
                    print("batch:%d, correct:%d"%(batch_size, batch_correct))
                
                correct += batch_correct
                total += batch_size
   
        print("%s Accuracy: %f"%(name, correct/total) )


def train_and_test():
        
    RUN_TRAINING = True
    n_big_loop = 10
    n_epochs = 5
    new_weight_file = "test_c01_new.pt"


    print("---------------")
    print("Load Dataset Step")
    
    load_dataset()
    
    print("---------------")
    print("Create Model Step")

    model = Net()
    #load_weights(WEIGHT_FILE, model)
    
    learning_rate = 0.01
    optimizer = lhcnn.SGD( lr=learning_rate)
    optimizer.setup(model)
    
    loss_fn = lhcnn.SoftmaxCrossEntropy()
    
    for bloop in range(n_big_loop):

        print("---------------")
        print("Train Model Step:",  bloop)
        
        if RUN_TRAINING:
            print("number of epochs:", n_epochs)
            training_loop(n_epochs, optimizer,  model, loss_fn, cifar2_train,prev_loops=bloop*n_epochs )
            
            #write_weights(model, new_weight_file)

        print("---------------")
        test_model(model, cifar2_train, cifar2_test )
        
    print("---------------")
    print("Finished")
    print("---------------")

def main_loop():

    print("---------------")
    print("Load Dataset Step")
    
    load_dataset()
    
    print("---------------")
    print("Load Model Step")

    model = Net()
    #load_weights(WEIGHT_FILE, model)

    print("---------------")
    print("Test Model Step")
    
    test_model(model, cifar2_train, cifar2_test, verbose = True )

    print("---------------")
    print("Finished")
    print("---------------")

if __name__ == '__main__':
    #main_loop()
    train_and_test()

