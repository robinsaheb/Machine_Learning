# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:58:09 2017

@author: sahebsingh
"""
"""Project Iris"""
##First we will VIsualize the data##


from matplotlib import pyplot as plt
#We load the data with load_ris from the Database

from sklearn.datasets import load_iris
import numpy as np

#load_iris returns an object with several fields.
data=load_iris()
features=data.data
feature_names=data.feature_names
target=data.target
target_names=data.target_names

fig,axes=plt.subplots(2,3)
pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
color_markers = [
        ('r', '>'),
        ('g', 'o'),
        ('b', 'x'),
        ]
for i, (p0, p1) in enumerate(pairs):
    ax = axes.flat[i]

    for t in range(3):
        # Use a different color/marker for each class `t`
        c,marker = color_markers[t]
        ax.scatter(features[target == t, p0], features[
                    target == t, p1], marker=marker, c=c)
    ax.set_xlabel(feature_names[p0])
    ax.set_ylabel(feature_names[p1])
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig('figure1.png')
#for t in range(3):
#    if t == 0:
#        c = 'r'
#        marker = '>'
#    elif t == 1:
#        c = 'g'
#        marker = 'o'
#    elif t == 2:
#        c = 'b'
#        marker = 'x'
#    plt.scatter(features[target == t,3],
#                features[target == t,2],
#                marker=marker,
#                c=c)    
labels=data.target_names[data.target]
plength=features[:,2]
is_setosa = (labels == 'setosa')
max_setosa=(plength[is_setosa]).max()
#print("This is max_seotsa %s"%max_setosa)
min_non_setosa=(plength[~is_setosa]).min()
#print("This is min Non setosa %s"%min_non_setosa)


features = features[~is_setosa]
labels = labels[~is_setosa]
is_virginica = (labels == 'virginica')

def fit_model(features, labels):
    '''Learn a simple threshold model'''
    best_acc = -1.0
    # Loop over all the features:
    for fi in range(features.shape[1]):
        thresh = features[:, fi].copy()
        # test all feature values in order:
        thresh.sort()
        for t in thresh:
            pred = (features[:, fi] > t)

            # Measure the accuracy of this 
            acc = (pred == labels).mean()

            rev_acc = (pred == ~labels).mean()
            if rev_acc > acc:
                acc = rev_acc
                reverse = True
            else:
                reverse = False
            if acc > best_acc:
                best_acc = acc
                best_fi = fi
                best_t = t
                best_reverse = reverse

    # A model is a threshold and an index
    return best_t, best_fi, best_reverse


# This function was called ``apply_model`` in the first edition
def predict(model, features):
    '''Apply a learned model'''
    # A model is a pair as returned by fit_model
    t, fi, reverse = model
    if reverse:
        return features[:, fi] <= t
    else:
        return features[:, fi] > t

def accuracy(features, labels, model):
    '''Compute the accuracy of the model'''
    preds = predict(model, features)
    return np.mean(preds == labels)

##Figure of the best Model##

COLOUR_FIGURE = False
t=1.6
t2 = 1.75

# Features to use: 3 & 2
f0, f1 = 3, 2

if COLOUR_FIGURE:
    area1c = (1., .8, .8)
    area2c = (.8, .8, 1.)
else:
    area1c = (1., 1, 1)
    area2c = (.7, .7, .7)

# Plot from 90% of smallest value to 110% of largest value
# (all feature values are positive, otherwise this would not work very well)

x0 = features[:, f0].min() * .9
x1 = features[:, f0].max() * 1.1

y0 = features[:, f1].min() * .9
y1 = features[:, f1].max() * 1.1

fig,ax = plt.subplots()
ax.fill_between([t, x1], [y0, y0], [y1, y1], color=area2c)
ax.fill_between([x0, t], [y0, y0], [y1, y1], color=area1c)
ax.plot([t, t], [y0, y1], 'k--', lw=2)
ax.plot([t2, t2], [y0, y1], 'k:', lw=2)
ax.scatter(features[is_virginica, f0],
            features[is_virginica, f1], c='b', marker='o', s=40)
ax.scatter(features[~is_virginica, f0],
            features[~is_virginica, f1], c='r', marker='x', s=40)
ax.set_title("The Line is the Decision Boundary\nThe Dotted Line Gives us same amount of Accuracy.")
ax.set_ylim(y0, y1)
ax.set_xlim(x0, x1)
ax.set_xlabel(feature_names[f0])
ax.set_ylabel(feature_names[f1])
fig.tight_layout()
fig.savefig('figure2.png')
    
# Split the data in two: testing and training
testing = np.tile([True, False], 50) # testing = [True,False,True,False,True,False...]

# Training is the negation of testing: i.e., datapoints not used for testing,
# will be used for training
training = ~testing

model = fit_model(features[training], is_virginica[training])
train_accuracy = accuracy(features[training], is_virginica[training], model)
test_accuracy = accuracy(features[testing], is_virginica[testing], model)

print('''\
Training accuracy was {0:.1%}.
Testing accuracy was {1:.1%} (N = {2}).
'''.format(train_accuracy, test_accuracy, testing.sum()))

correct=0.0
for ei in range(len(features)):
    training=np.ones(len(features),bool)
    training[ei]=False
    testing=~training
    model=fit_model(features[training],is_virginica[training])
    prediction=predict(model,features[testing])
    correct+=np.sum(prediction==is_virginica[testing])
acc=correct/float(len(features))
print("Accuracy : {0:.1%}".format(acc))







































