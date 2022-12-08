import numpy as np

'''
使用神经网络拟合数学函数
能用数学函数，或者确定的算法解决的事情，自然是最精确的，往往也是最省力的，但是，对于复杂的函数、系统，很难找到一条漂亮的数学函数或者算法，这时候可以考虑数据拟合，而神经网络可以做这个事情。
'''

class Dot(object):
    def forward(self, X, W):
        self.X=X
        self.W=W
        return np.dot(X, W)

    def backward(self, dout):
        return np.dot(dout, self.W.T), np.dot(self.X.T, dout)  # dX, dW

class Add(object):
    def forward(self, D, B):
        return D+B

    def backward(self, dout):
        return np.sum(dout, axis=0)  #纵向求和

class Affine(object):
    def __init__(self):
        self.dot=Dot()
        self.add=Add()

    def forward(self, X, W, B):
        T=self.dot.forward(X, W)
        return self.add.forward(T, B)

    def backward(self, dout):
        dB = self.add.backward(dout)
        dX, dW = self.dot.backward(dout)
        
        return dX, dW, dB

class Sigmoid(object):
    def forward(self, A):
        self.out=1/(1+np.exp(-A))
        return self.out

    def backward(self, dout):
        return dout*(1.0-self.out)*self.out

class Relu(object):
    def forward(self, A):
        self.mask=(A<=0)
        A[self.mask]=0     #会破坏原参数，但是这次使用之后后面就没用了，所以破坏就破坏了，还能省点空间
        return A

    def backward(self, dout):
        dout[self.mask]=0
        return dout

class SoftmaxWithLoss(object):
    def softmax(self, A):
        A=A-A.max(axis=1, keepdims=True)
        T=np.exp(A)
        return T/np.sum(T, axis=1, keepdims=True)

    def crossEntropyError(self, Z, Label):
        delta=0.000000001
        return -np.sum(np.log(Z+delta)*Label)/Z.shape[0]

    def forward(self, A, Label):
        self.Z=self.softmax(A)
        self.Label=Label
        return self.Z, self.crossEntropyError(self.Z, Label)

    def backward(self, dout=1):
        return (self.Z-self.Label)*dout/self.Z.shape[0]

class MSELoss(object):
    def forward(self, A, Label):
        self.A=A
        self.Label=Label
        # ∑( ∑(yi-xi)^2 )/batch_size
        return np.sum((A-Label)*(A-Label))/A.shape[0]    # A.shape[0] 为样本数，batch_size

    def backward(self, dout=1):
        return 2*(self.A-self.Label)*dout/self.A.shape[0]

class MyNN(object):
    def __init__(self):
        self.lr=0.01
        self.MAX_NUM=1000000

    def buildThreeLayerNet(self, input_dim, output_dim):
        N0=input_dim
        N1=10             #超参，为什么设置程这个数值，没啥理由，感觉
        N2=10
        N3=output_dim

        self.W1=np.random.randn(N0, N1)
        self.B1=np.random.randn(N1)
 
        self.W2=np.random.randn(N1, N2)
        self.B2=np.random.randn(N2)

        self.W3=np.random.randn(N2, N3)
        self.B3=np.random.randn(N3)

        self.affine1=Affine()
        self.activation1=Sigmoid()  #对于浅层的简单网络，相对于 Relu ，激活函数使用 Sigmoid 似乎表达能力更强一些
        self.affine2=Affine()
        self.activation2=Sigmoid()
        self.affine3=Affine()
        self.mseloss=MSELoss()

    def getMaxAbsValue(self, X):
        a=np.max(X)
        b=np.min(X)
        if b<0:
            b=-b

        ret=max(a,b)
        if ret==0:
            ret=0.0000001
        return ret

    def learn(self, X, Y):
        input_dim=len(X[0])
        output_dim=len(Y[0])

        self.MAX_X=self.getMaxAbsValue(X)
        self.MAX_Y=self.getMaxAbsValue(Y)
        X=np.array(X)/self.MAX_X
        Y=np.array(Y)/self.MAX_Y

        self.buildThreeLayerNet(input_dim, output_dim)

        for i in range(0, self.MAX_NUM):
            A1= self.affine1.forward(X, self.W1, self.B1)
            Z1=self.activation1.forward(A1)
            A2= self.affine2.forward(Z1, self.W2, self.B2)
            Z2=self.activation2.forward(A2)
            A3= self.affine3.forward(Z2, self.W3, self.B3)
            l = self.mseloss.forward(A3, Y)

            if i%10000==0:
                print(l)

                if l<0.0001:
                    #print(Y)
                    #print(A3)
                    #print(self.predict(X))
                    break

            dA3 = self.mseloss.backward(dout=1)
            dZ2, dW3, dB3 = self.affine3.backward(dA3)
            dA2 = self.activation2.backward(dZ2)
            dZ1, dW2, dB2 = self.affine2.backward(dA2)
            dA1 = self.activation1.backward(dZ1)
            dX, dW1, dB1 = self.affine1.backward(dA1)

            #根据梯度调整权重与偏置
            self.W1-=self.lr*dW1
            self.B1-=self.lr*dB1
            self.W2-=self.lr*dW2
            self.B2-=self.lr*dB2
            self.W3-=self.lr*dW3
            self.B3-=self.lr*dB3

    def predict(self, X):
        X=np.array(X)/self.MAX_X

        A1= self.affine1.forward(X, self.W1, self.B1)
        Z1=self.activation1.forward(A1)
        A2= self.affine2.forward(Z1, self.W2, self.B2)
        Z2=self.activation2.forward(A2)
        A3= self.affine3.forward(Z2, self.W3, self.B3)
        return A3*self.MAX_Y

class FitFx(object):
    def fx1(self, x):
        # f(x)=x-5
        return x-5

    def fx2(self, x):
        # f(x)=(x-5)^2
        return (x-5)**2

    def fx3(self, x):
        # f(x)=(x-5)^3+1
        return (x-5)**3

    def fx4(self, x):
        # f(x)=(x-5)^3+1
        return (x-5)**4

    def fx_xor(self, x1, x2):
        return x1^x2    # 异或

    def fit11(self, fx):
        print(fx)
        X=[[i] for i in range(0, 11)]
        print(X)
        Y=[[fx(a[0])] for a in X]
        print(Y)

        nn=MyNN()
        nn.learn(X,Y)
        print(nn.predict(X))
        #print(nn.predict([[1.2],[8.5]]))

    def fit21(self, fx):
        print(fx)
        X=[[d1,d2] for d1,d2 in [(0,0), (0,1), (1,0), (1,1)]]
        print(X)
        Y=[[fx(*a)] for a in X]
        print(Y)
        
        nn=MyNN()
        nn.learn(X,Y)
        print(nn.predict(X))


def main():
    m=FitFx()
    #m.fit11(m.fx1)
    m.fit11(m.fx2)
    #m.fit11(m.fx3)
    #m.fit11(m.fx4)

    #m.fit21(m.fx_xor)

if "__main__"==__name__:
    main()
