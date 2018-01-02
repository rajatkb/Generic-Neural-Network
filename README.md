# Generic-Neural-Network  !!work in progress
It contains my own variant of Neural Network implementation which is generic. So ya anyone who have used keras should be able to understand how it works. It allows you to add layers and then run the test

The general method goes like this

neural_net = gnn()

neural_net.add_layer(n , "activation") // n= number of nodes in the layer activation="relu" ,"sigmoid" , "tanh" (tanh not yet done)
neural_net.output_layer(1 , "sigmoid")

neural_net.train(X_train , Y_train , <br/>
                 epoch=1000000 , alpha=0.00001 , <br/>
                 drop_keep=1 , batches=10 , adam=[0.8,0.999,10e-8], l2=0 )
<br/>
Yn = neural_net.predict(X_train)
Yn = (Yn > 0.5)
print("Accuracy: " , accuracy(Y_train , Yn))

# Optimizations done
1. Batch norm (yet not implemented)
2. Mini Batch
3. Adaptive Momentum (weighted average + RMS prop)
4. Drop out
5. L2 regularization

And still it performs bad on simple datasets like breast cancer of sklearn. So either the models not right or i am not training it for enough time. 


