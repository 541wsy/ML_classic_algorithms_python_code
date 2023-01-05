import numpy as np


class LR(object):
    def __init__(self,fit_intercept=True):
        '''

        :param fit_intercept: 是否加入截距项，如果加入，需要对X第一列拼接上全1向量
        '''
        self.beta = None #线性回归参数
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        '''

        :param X: 样本矩阵，shape=[n_sample, n_feature]
        :param y: shape=[n_sample,1]
        :return:
        '''
        #当有截距项，需要在X左侧加上全1值
        if self.fit_intercept:
            X = np.hstack((np.ones_like(y.reshape((-1,1))), X))


        ##判断(XTX)是否可逆
        n_sample = X.shape[0]
        n_feature = X.shape[1]

        #当特征数量大于样本数量显然不可逆，因为XTX的shape是[n_feature,n_feature]
        if n_feature > n_sample:
            is_inv = False
        #进一步判断行列式
        elif np.linalg.det(np.matmul(X.T, X)) == 0:
            is_inv = False
        else:
            is_inv = True

        #当可逆
        if is_inv:
            self.beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        #当不可逆
        else:
            u,s,vt = np.linalg.svd(X) #SVD分解,注意这里的s是向量，表示奇异值，
                                    # 后续进行矩阵乘法需要转成矩阵
                                    #还有一个坑！这里的v是vT，而不是v，后续要进行转置
            if len(np.nonzero(s)[0]) == X.shape[0]:
                sigma_inv_vector = 1 / s #奇异值倒数向量
            else:#当出现0奇异值，1/s会报错
                n_nonzero = len(np.nonzero(s)[0])
                s_nonzero = s[:n_nonzero]
                s_inv = 1 / s #对角阵的伪逆
                zeros = np.zeros((n_feature - len(s_inv)))
                sigma_inv_vector = np.hstack((s_inv,zeros))
            ##奇异值倒数向量sigma_inv_vector转成矩阵sigma_inv
            sigma_inv_diag = np.diag(sigma_inv_vector) #sigma_inv的对角部分
            if X.shape[0] == X.shape[1]: #当sigma方阵
                sigma_inv = sigma_inv_diag
            elif X.shape[0] > X.shape[1]: #当sigma是竖的矩形
                sigma_zeros = np.zeros((X.shape[1],(X.shape[0] - X.shape[1])))
                sigma_inv = np.hstack((sigma_inv_diag, sigma_zeros))
            else:#当sigma是横的矩形
                sigma_zeros = np.zeros(((X.shape[1] - X.shape[0]),X.shape[0]))
                sigma_inv = np.vstack((sigma_inv_diag, sigma_zeros))

            self.beta = vt.T @ sigma_inv @ u.T @ y

        self.beta = self.beta.reshape((-1,1))

    def predict(self, X):
        '''

        :param X: 测试集，shape=[n_sample, n_feature]
        :return: y_predict,shape=[n_sample,1]
        '''
        if X.shape[1] != self.beta.shape[0]:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        y_predict = X.dot(self.beta)
        return y_predict


def cal_mse(y_predict, y_true):
    assert y_predict.ndim == y_true.ndim, 'y_predict和y_true需要维度相同'

    return np.mean((y_predict - y_true) ** 2)


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    print('=====实验1：可逆情形=====')
    #实验1：可逆的XTX
    X,y, coef = make_regression(n_samples=100, n_features=10, n_informative=10, coef=True, random_state=2022)
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2022)

    print('=====My Linear Regression=====')
    #实现的LR
    lr = LR(fit_intercept=True)
    lr.fit(X_train,y_train)

    y_train_predict = lr.predict(X_train)
    y_test_predict = lr.predict(X_test)

    ##计算训练集MSE，测试集MSE
    print('MSE in training set:%.4f'%(cal_mse(y_train_predict, y_train.reshape((-1,1)))))
    print('MSE in testing set:%.4f' % (cal_mse(y_test_predict, y_test.reshape((-1,1)))))

    #sklearn的LR
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    lr_sk = LinearRegression()
    lr_sk.fit(X_train,y_train)

    y_train_predict_sk = lr_sk.predict(X_train)
    y_test_predict_sk = lr_sk.predict(X_test)

    print('=====Sklearn Linear Regression=====')
    ##计算训练集MSE，测试集MSE
    print('MSE in training set:%.4f'%(mean_squared_error(y_train, y_train_predict_sk)))
    print('MSE in testing set:%.4f' % (mean_squared_error(y_test, y_test_predict_sk)))

    print('=====实验2：不可逆情形=====')
    # 实验2：不可逆的XTX
    X, y, coef = make_regression(n_samples=10, n_features=100, n_informative=100, coef=True, random_state=2022)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2022)

    # 实现的LR
    lr = LR(fit_intercept=True)
    lr.fit(X_train, y_train)

    y_train_predict = lr.predict(X_train)
    y_test_predict = lr.predict(X_test)

    print('=====My Linear Regression=====')
    ##计算训练集MSE，测试集MSE
    print('MSE in training set:%.4f' % (cal_mse(y_train_predict, y_train.reshape((-1, 1)))))
    print('MSE in testing set:%.4f' % (cal_mse(y_test_predict, y_test.reshape((-1, 1)))))

    # sklearn的LR
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    lr_sk = LinearRegression()
    lr_sk.fit(X_train, y_train)

    y_train_predict_sk = lr_sk.predict(X_train)
    y_test_predict_sk = lr_sk.predict(X_test)

    print('=====Sklearn Linear Regression=====')
    ##计算训练集MSE，测试集MSE
    print('MSE in training set:%.4f' % (mean_squared_error(y_train, y_train_predict_sk)))
    print('MSE in testing set:%.4f' % (mean_squared_error(y_test, y_test_predict_sk)))

    print('SVD的解是当OLS解不唯一时，最小二范数解')
    print('SVD LR coef 2-norm:%.4f'%np.linalg.norm(lr.beta))
    print('sklearn LR coef 2-norm:%.4f' % np.linalg.norm(np.hstack((lr_sk.coef_, lr_sk.intercept_))))
