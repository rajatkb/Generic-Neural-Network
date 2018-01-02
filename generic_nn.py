import numpy as np


class generic_nn:
    '''
    Forward Prop:
        In order to start using the model. First create the layer by add layer function.
        The add_layer() function just adds number of neurons and the activation in that level 
        as a dictionary to the layers list
        
        for each layer the forward_prop() iterates while running forward_prop_unit() over it
        
        for each layer the backward_prop() iterates while running backward_prop_unit for each layer
            
    '''
    
    def __init__(self):
        '''
        Var:
            layers[] = Conatains the cache and weights of each neuron
        function:
            initializes the weight values using Xavier Initialization
        '''
        self.layers = []
        self.layer_count =0
        self.activaton={"sigmoid":self.sigmoid_activation , 
                        "tanh":self.tanh_activation ,
                        "relu":self.relu_activation , 
                        "softmax":self.softmax_activation}
        self.activation_backprop={"relu": self.relu_differentiation,
                                  "sigmoid":self.sigmoid_differentiation
                          }
        self.drop_keep=0.5 
        self.epsilon = 0.9
        self.alpha = 0.01
        self.beta1=0.9
        self.beta2 = 0.999
        # for graphing purpose
        
        
    def add_layer(self , neurons , activation="relu"):
        self.layers.append({"node_count":neurons , "activation":activation})  
        self.layer_count+=1
        
    def output_layer(self , neurons , activation="softmax"):    
        self.layers.append({"node_count":neurons , "activation":activation})
        self.layer_count+=1
    
    def initialize_weights(self ,input_layer):
        self.layers[0]["W"]=np.random.randn(self.layers[0]["node_count"],input_layer)*np.sqrt(2/input_layer)*0.01
        self.layers[0]["b"]=np.zeros([self.layers[0]["node_count"],1])
        self.layers[0]["vdW"]=np.zeros([self.layers[0]["node_count"],input_layer])
        self.layers[0]["vdb"]=np.zeros([self.layers[0]["node_count"],1])
        self.layers[0]["sdW"]=np.zeros([self.layers[0]["node_count"],input_layer])
        self.layers[0]["sdb"]=np.zeros([self.layers[0]["node_count"],1])
#       self.layers[0]["gamma"] = np.random.randn(self.layers[0]["node_count"],1)*np.sqrt(2/input_layer)*0.01
#        self.layers[0]["beta"] = np.random.randn(self.layers[0]["node_count"],1)*np.sqrt(2/input_layer)*0.01
        input_layer = self.layers[0]["node_count"]
        for i in range(1 , len(self.layers)):
            self.layers[i]["W"]=np.random.randn(self.layers[i]["node_count"],input_layer)*np.sqrt(2/input_layer)*0.01
            self.layers[i]["b"]=np.zeros([self.layers[i]["node_count"],1])
            self.layers[i]["vdW"]=np.zeros([self.layers[i]["node_count"],input_layer])
            self.layers[i]["vdb"]=np.zeros([self.layers[i]["node_count"],1])
            self.layers[i]["sdW"]=np.zeros([self.layers[i]["node_count"],input_layer])
            self.layers[i]["sdb"]=np.zeros([self.layers[i]["node_count"],1])
#            self.layers[i]["gamma"] = np.random.randn(self.layers[i]["node_count"],1)*np.sqrt(2/input_layer)*0.01
#            self.layers[i]["beta"] = np.random.randn(self.layers[i]["node_count"],1)*np.sqrt(2/input_layer)*0.01
            input_layer = self.layers[i]["node_count"]
        return self.layers  
    
    def relu_activation(self , Z):
        return Z * (Z >= 0)
    
    def relu_differentiation(self , Anext, dAl):
        return np.ones(Anext.shape)*(Anext > 0)*dAl 
    
    
    def sigmoid_activation(self , Z):    
        return 1/(1+ np.exp(-Z))
    
    def sigmoid_differentiation(self,Anext, dAl):
        return Anext*(1-Anext)*dAl
    
    def tanh_activation(self , Z):
        return np.tanh(Z)
    
    def softmax_activation(self , Z):
        Xd=np.exp(Z)
        return Xd/np.sum(Xd , axis=0)
        
    def normalize_epsilon(self , X ):
        X_Sub_mean = X - np.mean(X , axis=1).reshape(X.shape[0], 1)
        return (X_Sub_mean / np.sqrt(self.epsilon + np.sum(X_Sub_mean**2 ,axis=1).reshape(X.shape[0],1)))

    
    def forward_prop_unit(self , Aprev ,depth):
        '''
        Arg: 
            Activation of previous layer : Aprev
            the layers depth from 0 : depth
        fun:
            does batch normalization
            and stores drop_out , Aprev , Z in layer dict
        return:
            Anext
        '''
        drop_keep = self.drop_keep
        
        # coputing Anext and Z
        Z = np.dot(self.layers[depth]["W"] , Aprev)+self.layers[depth]["b"]
        #batch normaization -- will do
#        Z=self.normalize_epsilon(Z)
#        Z = self.layers[depth]["gamma"]*Z + self.layers[depth]["beta"]
        #computing
        Anext= self.activaton[self.layers[depth]["activation"]](Z)
        #saving Z and Aprev
        self.layers[depth]["Z"] = Z
        self.layers[depth]["Aprev"] = Aprev
        # performing drop out
        if(depth != (self.layer_count-1)):
            drop_out = np.random.random(Anext.shape)  < drop_keep
            Anext = Anext * drop_out
            Anext = Anext / drop_keep
            self.layers[depth]["drop_out"]=drop_out
        # scaling the undropped neurons and saving the boolean matrix for drop out
        self.layers[depth]["Anext"]=Anext
        
        return Anext
    
    
    def forward_prop(self , X  ):
        # forward propogation
        A = self.forward_prop_unit(X , 0 )#first prop of layer 1
        for l in range(1 , self.layer_count):
            A = self.forward_prop_unit(A , l )
        return A
    
    def adam_optimization(self , depth,dW,db):
        self.layers[depth]["vdW"] = (self.beta1 * self.layers[depth]["vdW"] + (1-self.beta1)*dW)
        self.layers[depth]["vdb"] = (self.beta1 * self.layers[depth]["vdb"] + (1-self.beta1)*db)
        self.layers[depth]["sdW"] = (self.beta2 * self.layers[depth]["sdW"] + (1-self.beta2)*np.square(dW))
        self.layers[depth]["sdb"] = (self.beta2 * self.layers[depth]["sdb"] + (1-self.beta2)*np.square(db))
        dW = self.layers[depth]["vdW"] / (np.sqrt(self.layers[depth]["sdW"]) + self.epsilon)
        dW = self.layers[depth]["vdb"] / (np.sqrt(self.layers[depth]["sdb"]) + self.epsilon)
        return (dW,db)
    
    def l2_regularization(self , depth , dW , db):
        return (dW+((self.l2*self.layers[depth]["W"])/self.m),(db+(self.l2*self.layers[depth]["b"])/self.m))
    
    def l2_cost(self):
        s=0
        for l in self.layers:
            s+=np.sum(np.square(l["W"]))
        return (self.l2 * s)/(2*self.m) 
        
    
    def backward_prop_unit(self , dAprev , depth):
        '''
            Arg:
                dA of earlier layers when moving backward respect to loss function : dAprev
                depth denotes which layer we are dealing with index 0..n : depth
            fun:
                calculates the derivatives and updates the weights in each induvidual layer
            return:
                dAnext the dA that need to be passed to the next unit
        '''
        # accouting dropped unit
        if(depth != (self.layer_count-1)):
            dAprev = dAprev * self.layers[depth]["drop_out"]
        # getting dZ computed in respect to the dAprev recieved
        dZ = self.activation_backprop[self.layers[depth]["activation"]](self.layers[depth]["Anext"],dAprev)
        dW = (np.dot(dZ,self.layers[depth]["Aprev"].T))/self.m
        db = np.mean(dZ , axis=1).reshape(dZ.shape[0],1) 
        # the reshape is just a fix for what mean operation does to your matrix
        dAnext = np.dot(self.layers[depth]["W"].T , dZ)
        #adam optimization
        dW,db = self.adam_optimization(depth , dW , db)
        #l2 regularization
        dW,db = self.l2_regularization(depth , dW , db)
        # update routines
        self.layers[depth]["W"] -= self.alpha * dW
        self.layers[depth]["b"] -= self.alpha * db
        return dAnext
    
    def backward_prop(self , dALoss):
        #backward propogation
        for l in reversed(range(self.layer_count)):
            dALoss = self.backward_prop_unit(dALoss , l)
        return dALoss
    
    def calculate_cost(self , Y , Al , function):
        if function == "relu":
            return np.mean((Y-Al)**2 , axis=1) +self.l2_cost()
        if function == "sigmoid":
            return np.mean(Y*np.log(Al) + (1-Y)*np.log(1-Al) , axis=1) +self.l2_cost()
        if function == "softmax":
            return np.mean(Y*np.log(Al) ,axis=1) +self.l2_cost()
    
    def calculate_cost_gradient(self , Y , Al , function):
        if function == "relu":
            return -2*(Y - Al)
        if function == "sigmoid":
            return -(np.divide(Y , Al) - np.divide(1-Y , 1-Al))

        
    
    def train(self , X , Y ,epoch=1000 , alpha=0.01 , drop_keep=0.5 , batches= 5 , adam=[0.9 , 0.999 , 10e-8] , l2=0.5,show_graph=True):
        
        self.m = X.shape[0] # number of total data
        self.drop_keep = drop_keep
        self.alpha = alpha
        self.batches = batches
        self.beta1 = adam[0]
        self.beta2 = adam[1]
        self.epsilon = adam[2]
        self.l2=l2
        batch_size = int(self.m / self.batches)
        
        self.initialize_weights(self.m) #initializing the weights
        
        for i in range(epoch):
            for b in range(batches):
                Al = self.forward_prop(X[:,batch_size*b: batch_size*(b+1)])
                cost = self.calculate_cost(Y[:,batch_size*b: batch_size*(b+1)] , Al , self.layers[-1]["activation"])
                print("\nThe cost : " ,cost ,"iteration:",i)
                dALoss = self.calculate_cost_gradient(Y[:,batch_size*b: batch_size*(b+1)] , Al , self.layers[-1]["activation"])
                dXLoss = self.backward_prop(dALoss)
            if (batches == 1 or batches == self.m):
                continue
            Al = self.forward_prop(X[:,batch_size*batches: ])
            cost = self.calculate_cost(Y[:,batch_size*batches: ] , Al , self.layers[-1]["activation"])
            print("\nThe cost : " ,cost )
            dALoss = self.calculate_cost_gradient(Y[:,batch_size*batches: ] , Al , self.layers[-1]["activation"])
            dXLoss = self.backward_prop(dALoss)    
                
        return None
    
    def predict(self , Xn):
        A=Xn
        for depth in range(self.layer_count):
            Z = np.dot(self.layers[depth]["W"] , A)+self.layers[depth]["b"]
            A= self.activaton[self.layers[depth]["activation"]](Z)
        return A