
import lhcnn
import numpy as np
#from matplotlib import pyplot as plt
        
t_c = np.asarray([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21])
t_c = t_c.reshape(11, 1)

t_u = np.asarray([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]) * 0.1
t_u = t_u.reshape(11, 1)


def training_loop( n_epochs,  optimizer,  model,  loss_fn,  t_u_val,  t_c_val):
    print("start training_loop")
    
    for epoch in range(n_epochs):
        t_p_val = model(t_u_val)
        loss_val = loss_fn(t_p_val,  t_c_val)

        model.cleargrads()
        
        loss_val.backward()
        optimizer.update()
        
        #rint( "epoch %d, loss: %f"%(epoch,  loss_val.item()) )
        print( "epoch %d, loss: %f"%(epoch,  loss_val.data) )
        
        if epoch % 100 == 0:
            print( "epoch %d, loss: %f"%(epoch,  loss_val.data) )
            print("weight =>", model.W)
            print("bias =>",  model.b)

    print("training_loop finished")
    print("weight => ", model.W, "bias =>",  model.b)
    print("-----------------------")

def test_linear1():

    model = lhcnn.Linear(1, 1)

    optimizer = lhcnn.SGD( lr=1e-2)
    optimizer.setup(model)

    loss_fn = lhcnn.MeanSquaredErrorFn()
    training_loop(2000, optimizer, model, loss_fn, t_u, t_c)


test_linear1()
#test_nn()

print("-----------------------")
