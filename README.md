# 機器學習概論 HW1 Report

# Part1 
## Split the train and test data
#### Using pandas to finish csv files editing

```python=
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

    training_set.to_csv('training_set.csv')
    testing_set.to_csv('testing_set.csv')

    return training_set, testing_set
```


# Part2
## Seperate each feature into a group with Label
#### split into x and y
```python=
x_dataset = dataset[:, 1:] #features
y_dataset = dataset[:, 0] # Label
```
#### classify into labels
```python=
for i in range(len(y_dataset)):
    if(y_dataset[i] == 1):
        type1.append(x_dataset[i])

    elif(y_dataset[i] == 2):
        type2.append(x_dataset[i])

    elif(y_dataset[i] == 3):
        type3.append(x_dataset[i])
```
#### For each feature, it has index for its label
```python=
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
```

## Calculate mean, standard and normal distribution
#### Get 13 features mean, standard and normal distribution with numpy and synic.stats
![](https://i.imgur.com/8c0cJw7.png)
```python=
for i in range(13):
    type1_mean.append(np.mean(type1[i]))
    type2_mean.append(np.mean(type2[i]))
    type3_mean.append(np.mean(type3[i]))
```

```python=
for i in range(13):
    type1_std.append(np.std(type1[i]))
    type2_std.append(np.std(type2[i]))
    type3_std.append(np.std(type3[i]))
```

```python=
    for i in range(13):
        type1_norm.append(st.norm(type1_mean[i], type1_std[i]))
        type2_norm.append(st.norm(type2_mean[i], type2_std[i]))
        type3_norm.append(st.norm(type3_mean[i], type3_std[i]))
```
## Prior
#### The distrubution of Labels will be the prior distribution
```python
prior[0] = len(type1) / len(y_dataset) #0.33
prior[1] = len(type2) / len(y_dataset) #0.42
prior[2] = len(type3) / len(y_dataset) #0.24
```

## Likelihood
![](https://i.imgur.com/ECCPtTL.png)

```python=
likelihood = sc.integrate.quad(dis[label][i].pdf, data[i+1], data[i+1]+delta )[0]
```

## Posterior
![](https://i.imgur.com/z8Aa3CI.png)

```python=
post[label] = 1. * prior[label]
```
```python=
post[label] = post[label]*likelihood
```

## MAP

```python=
predict = np.argmax(post)
    total+=1
    if predict == (data[0] -1):
        correct += 1
    else:
        pass
print('accuracy: ', correct/total)
```

# Part3
## Dimension 13 features into 2 & 3 features

```python=
X_p = pca2.fit(X).transform(X)
```

```python=
X_p = pca3.fit(X).transform(X)
```

## Visualization

```python=
for c, i, target_name, m in zip('rbg', labels, types, markers):
        plt2D.scatter(X_p[y==i, 0], X_p[y==i, 1], c=c, label=target_name, marker=m)
```

```python=
for c, i, target_name, m in zip('rgb', labels, types, markers):
        plt3D.scatter(X_p[y==i, 0], X_p[y==i, 1], X_p[y==i, 2], c=c, label=target_name, marker=m)
```
![](https://i.imgur.com/mPU51Ct.png)


# Part4 
## Discussion with different Prior
#### First, I just try serveral cases to observe the difference with each of them.
#### The more labels approaches to 0 , the worse accuracy will be.
#### In the other hand, be more close to the training distribution of 3 labels, accuracy will be more higher.
```python=
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
```

### Discover most important factors
#### Then I check critical roles in 13 features, Proline & Magnesium are 2 highest metric affect results.
```python=
pd.DataFrame(pca2.components_,columns=X.columns,index = ['Transform_feature_1','Transform_feature_2'])
```
```csvpreview {header="true"}
,Alcohol, Malic acid,Ash, Alcalinity of ash,Magnesium,Total phenols,Flavanoids,Non Flavonoid phenols,Proanthocyanins,Color intensit,Hue,OD280/OD315 of diluted wines,Proline
Transform_feature_1,0.001404923002591525,-0.0009400548719844526,0.000151535166501538,-0.004549820197070899,0.013375340813766888,0.001153967382729312,0.0018712072011085584,-0.00014111648252168802,0.0009697094997752541,0.0013830233929473879,0.0002716398837700701,0.0009338737832450454,0.9998944279022249
Transform_feature_2,-0.002664669843727335,-0.006757898699927968,0.0033694597743166416,-0.0028190716942057656,0.999672677593252,-0.0015390253776459933,0.004565924225872175,-0.0030373987840452235,0.008213294083240043,-0.01723057055356069,0.0023053808129246817,0.0011400228694411635,-0.013381342992320381

```

#### Proline ,Magnesium, and Alcalinity of ash are 3 highest metric
```python=
pd.DataFrame(pca3.components_,columns=X.columns,index = ['Transform_feature_1','Transform_feature_2', 'Transform_feature_3'])
```
```csvpreview {header="true"}
,Alcohol, Malic acid,Ash, Alcalinity of ash,Magnesium,Total phenols,Flavanoids,Non Flavonoid phenols,Proanthocyanins,Color intensit,Hue,OD280/OD315 of diluted wines,Proline
Transform_feature_1,0.001404923002591525,-0.0009400548719844526,0.000151535166501538,-0.004549820197070899,0.013375340813766888,0.001153967382729312,0.0018712072011085584,-0.00014111648252168802,0.0009697094997752541,0.0013830233929473879,0.0002716398837700701,0.0009338737832450454,0.9998944279022249
Transform_feature_2,-0.002664669843727335,-0.006757898699927968,0.0033694597743166416,-0.0028190716942057656,0.999672677593252,-0.0015390253776459933,0.004565924225872175,-0.0030373987840452235,0.008213294083240043,-0.01723057055356069,0.0023053808129246817,0.0011400228694411635,-0.013381342992320381
Transform_feature_3,0.04599422668581956,0.1387077164205851,0.06079330103716166,0.8480090762003314,0.012377882087021046,-0.01644477134059868,-0.06718036318742247,0.013756770960772543,-0.011182004251728099,0.493429330861757,-0.0342997567959337,-0.07693143446632932,0.003305848466985506

```


#### Just use three most important features to calculate MAP
#### accuracy = 0.81
```python=
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
```

#### We still have over 80% accuracy