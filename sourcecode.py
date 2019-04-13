import json
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


source ='data\dataset.json'

class DataSet:
    
    
    def __init__(self,source):
        
        self.categories=[]
        
        #loading the json dataset 
        with open(source,encoding='utf-8') as f:
            self.data=json.load(f)

        #lsiting  categories types in the dataset
        for c in self.data:
            if c['category'] not in self.categories:
                self.categories.append(c['category'])
        
        #calculating number of times each category occurrs
        
        #initializing each category to zero
        self.categories_freq={c:0 for c in self.categories}
        
        for c in self.data:
            self.categories_freq[c['category']] += 1
      
    #spliting test data inot train and test daatest
    def SplitDataset(self): 
        #shuffling the data at random
        np.random.shuffle(self.data)
        
        #spliting training and testing at 7:3 ratio
        xtrain_head=[h['headline'] for h in self.data[:140597]]
        xtrain_sd = [k['short_description'] for k in self.data[:140597]]
        xtrain = [ xtrain_head[i] +' ' + xtrain_sd[i] for i in range(len(xtrain_head)) ]
        ytrain=[c['category'] for c in self.data[:140597]]
        
        xtest_head=[h['headline'] for h in self.data[140597:]]
        xtest_sd=[k['short_description'] for k in self.data[140597:]]
        xtest = [ xtest_head[i] +' ' + xtest_sd[i] for i in range(len(xtest_head)) ]

        ytest=[c['category'] for c in self.data[140597:]]
        
        return xtrain,ytrain,xtest,ytest
        
        
        

    #cleaning text from uncessery characters
    def CleanText(self,dataset):
            
        sentence=[]
        for s in dataset:
            #removing charcacters
            se=re.sub('[\W]',' ',s)
            #remove numeric values
            se=re.sub('[0-9]','',se)
            se=re.sub('[_]','',se)
            
            sentence.append(se.lower())
            
        return sentence
    
    #mapping category to a  number
    def labels(self):
        label={}
        for i in range(len(self.categories)):
            label.update({self.categories[i]:i})
        return label
    
    
   
    
        
 # main()       

#getting data for the models
dataset=DataSet(source)
x_train,y_train,x_test,y_test=dataset.SplitDataset()
train_x=dataset.CleanText(x_train)
test_x=dataset.CleanText(x_test)

labels=dataset.labels()

train_y=[labels[a]for a in y_train ]
test_y=[labels[a]for a in y_test]



#using tfidf approach 

vectorizer=TfidfVectorizer(stop_words='english',norm='l2',ngram_range=(1,1)  ) #max_features=n
train_data=vectorizer.fit_transform(x_train) #train_x




#opening datsetresult.json
with open('datasetresult.json',encoding='utf-8') as d:
    dataresult=json.load(d)
    
d.close()

predict_data=[]
data_id=[]

for h in dataresult:
    predict_data.append(h['headline'])
    data_id.append(h['id'])

#data to be predicted    
test_data=vectorizer.transform(x_test) #test_x

data_to_predict=vectorizer.transform(predict_data)
   
        
#training and prediction using Naive bayes algorithm

clf=MultinomialNB(alpha=0.1) #aplha = 0.5 52%acc #aloha = 0 acc=47%
clf.fit(train_data,train_y)

#predicting the test data
test_predictions=clf.predict(test_data)     
#predicting datasetresult.json data
predictions=clf.predict(data_to_predict)


#printing test data predictions accuracy

print("{0:.2f}".format(accuracy_score(test_predictions,test_y)*100))

# wririting to the json file

classes=[]
for i in range(len(predictions)):
    for key,value in labels.items():
        if value == predictions[i]:    
            classes.append(key)
        

        

data={}
data['predictions']=[]
for i in range(len(predictions)):
    
    data['predictions'].append({"id":data_id[i],"category":classes[i]})
    
    
with open ("result\predicted.json",'w',encoding='utf-8') as out:
      
        json.dump(data,out,indent=4	)

out.close()        
        
            

