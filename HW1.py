from cgi import test
from tkinter.messagebox import NO
import pandas as pd
import numpy as np
import random
import scipy as sc
import scipy.stats as st
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


prior = [1., 1., 1.]

def read_and_split(filename):
    file = pd.read_csv(filename, header=None)
    file.columns = ['Label','Alcohol',' Malic acid','Ash',' Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Non Flavonoid phenols','Proanthocyanins','Color intensit','Hue','OD280/OD315 of diluted wines','Proline']
    
    type1_filt = (file['Label'] == 1)
    type2_filt = (file['Label'] == 2)
    type3_filt = (file['Label'] == 3)
    
    type1 = file.loc[type1_filt]
    type2 = file.loc[type2_filt]
    type3 = file.loc[type3_filt]

    
    type1_test = type1.sample(n=18, frac=None ,random_state=200)
    type1_train = type1.drop(type1_test.index)
    type2_test = type2.sample(n=18, frac=None ,random_state=200)
    type2_train = type2.drop(type2_test.index)
    type3_test = type3.sample(n=18, frac=None ,random_state=200)
    type3_train = type3.drop(type3_test.index)

    training_set = pd.concat([type1_train, type2_train, type3_train])
    testing_set = pd.concat([type1_test, type2_test, type3_test])

    #print(testing_set.shape)
    #print(training_set.shape)

    training_set.to_csv('training_set.csv')
    testing_set.to_csv('testing_set.csv')

    return training_set, testing_set
    
   # print(file.head(5))

 
def Preprocess(data):
    dataset = data.to_numpy()
    x_dataset = dataset[:, 1:] #features
    y_dataset = dataset[:, 0] # Label
    #print(x_dataset)

    type1 = []
    type2 = []
    type3 = []
    for i in range(len(y_dataset)):
        if(y_dataset[i] == 1):
            type1.append(x_dataset[i])

        elif(y_dataset[i] == 2):
            type2.append(x_dataset[i])

        elif(y_dataset[i] == 3):
            type3.append(x_dataset[i])

    #Different case of Prior
    # acc = 1.0 
    prior[0] = len(type1) / len(y_dataset) #0.33
    prior[1] = len(type2) / len(y_dataset) #0.42
    prior[2] = len(type3) / len(y_dataset) #0.24
    
    #acc = 0.96
    """
    prior[0] = 0.05
    prior[1] = 0.05
    prior[2] = 0.9
    """
    #acc = 0.87
    """
    prior[0] = 0.9
    prior[1] = 0.09999999999
    prior[2] = 0.00000000001
    """
    #acc = 0.57
    """
    prior[0] = 0.000000000005
    prior[1] = 0.99999999999
    prior[2] = 0.000000000005
    """
    #acc = 0.33
    """
    prior[0] = 0.0
    prior[1] = 0.1
    prior[2] = 0.0
    """
    
    type1_ft = np.zeros(shape=[13, len(type1)])
    type2_ft = np.zeros(shape=[13, len(type2)])
    type3_ft = np.zeros(shape=[13, len(type3)])
    for idx, ele in enumerate(x_dataset[0:len(type1)]):
        for i in range(13):
            type1_ft[i][idx] = x_dataset[idx][i]
    for idx, ele in enumerate(x_dataset[len(type1):len(type1)+len(type2)]):
        for i in range(13):
            type2_ft[i][idx] = x_dataset[len(type1)+idx][i]
    for idx, ele in enumerate(x_dataset[len(type1)+len(type2):len(y_dataset)]):
        for i in range(13):
            type3_ft[i][idx] = x_dataset[len(type1)+len(type2)+idx][i]
    #print(type1_ft.shape, type2_ft.shape, type3_ft.shape)

    return type1_ft, type2_ft, type3_ft

def get_mean_and_std(type1, type2, type3):
    type1_mean = []
    type2_mean = []
    type3_mean = []
    for i in range(13):
        type1_mean.append(np.mean(type1[i]))
        type2_mean.append(np.mean(type2[i]))
        type3_mean.append(np.mean(type3[i]))

    type1_std = []
    type2_std = []
    type3_std = []
    for i in range(13):
        type1_std.append(np.std(type1[i]))
        type2_std.append(np.std(type2[i]))
        type3_std.append(np.std(type3[i]))

    nor_dis = []
    type1_norm = []
    type2_norm = []
    type3_norm = []
    for i in range(13):
        type1_norm.append(st.norm(type1_mean[i], type1_std[i]))
        type2_norm.append(st.norm(type2_mean[i], type2_std[i]))
        type3_norm.append(st.norm(type3_mean[i], type3_std[i]))
    nor_dis.append(type1_norm)
    nor_dis.append(type2_norm)
    nor_dis.append(type3_norm)
    
    return nor_dis


def MAP(data, dis):
    test_data = data.to_numpy()
    np.random.shuffle(test_data)
         
    total = 0
    correct = 0
    delta = 1e-6
    for data in test_data:
        #print(data[0])
        post = [1., 1., 1.]
        for label in range(3):
            post[label] = 1. * prior[label]
            # Post = prior * likelihood

            # Use most important 3 factors as feature to calculate posterior
            # acc = 0.81
            '''
            for i in range(3):
                if i == 0:
                    # 13th feature
                    i = 13-1
                elif i == 1:
                    # 5th feature
                    i = 5-1
                else:
                    # 4th feature
                    i = 4-1
                likelihood = sc.integrate.quad(dis[label][i].pdf, data[i+1], data[i+1]+delta )[0]
                post[label] = post[label]*likelihood
            '''
            
            # acc = 1.0
            for i in range(13):
                likelihood = sc.integrate.quad(dis[label][i].pdf, data[i+1], data[i+1]+delta )[0]
                post[label] = post[label]*likelihood
            
        predict = np.argmax(post)
        total+=1
        if predict == (data[0] -1):
            correct += 1
        else:
            pass
    print('accuracy: ', correct/total)
            
            
def plot_it(data):
    #test_data = data.to_numpy()
    #data = pd.DataFrame(data)
    #print(data)
    X = data.drop('Label', axis=1)
    y = data['Label']
    #print(y)

    fig = plt.figure(figsize=(12,6))
    markers = ['s', 'x', 'o']
    types = ['Label_1', 'Label_2', 'Label_3']
    labels = [1, 2, 3]

    '''Transform to 2 Dimension'''
    pca2 = PCA(n_components=2)
    # transform to 2 dimension
    X_p = pca2.fit(X).transform(X)
    plt2D = fig.add_subplot(1,2,1)
    for c, i, target_name, m in zip('rbg', labels, types, markers):
        plt2D.scatter(X_p[y==i, 0], X_p[y==i, 1], c=c, label=target_name, marker=m)
    plt2D.set_xlabel('Transform_feature_1')
    plt2D.set_ylabel('Transform_feature_2')
    plt.legend(loc='upper right')

    # Display most important 2 factores
    # Proline & Magnesium are important
    pd.DataFrame(pca2.components_,columns=X.columns,index = ['Transform_feature_1','Transform_feature_2'])

    '''Transform to 3 dimension'''
    plt3D = fig.add_subplot(1,2,2, projection='3d')
    pca3 = PCA(n_components=3)
    X_p = pca3.fit(X).transform(X)

    for c, i, target_name, m in zip('rgb', labels, types, markers):
        plt3D.scatter(X_p[y==i, 0], X_p[y==i, 1], X_p[y==i, 2], c=c, label=target_name, marker=m)

    plt3D.set_xlabel('Transform_feature_1')
    plt3D.set_ylabel('Transform_feature_2')
    plt3D.set_zlabel('Transfrom_feature_3')
    plt.legend(loc='upper right')

    # Display most important 2 factores
    # Proline ,Magnesium, and Alcalinity of ash are important
    pd.DataFrame(pca3.components_,columns=X.columns,index = ['Transform_feature_1','Transform_feature_2', 'Transform_feature_3'])

    plt.savefig('./report/visualize.png', dpi=300)
    plt.show()
    





def main():
    # Part1
    training_set, testing_test = read_and_split("Wine.csv")
    type1_ft, type2_ft, type3_ft =  Preprocess(training_set)

    # Part2
    nor_dis = get_mean_and_std(type1_ft, type2_ft, type3_ft)
    MAP(testing_test, nor_dis)

    # Part3
    plot_it(testing_test)

if __name__ == "__main__":
    main()
