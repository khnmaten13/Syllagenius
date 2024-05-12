
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import phrase_detect
import pickle


X_train,x,y,test_data_features,Y_test=phrase_detect.dataBuilder()
filename='final_model.sav'
print("Model loaded!!")
#Loading pickle file
loaded_model=pickle.load(open(filename, 'rb'))
print("Evaluating!!")
#Evaluating unlabelled data
eval1=pd.read_csv("eval_data.csv")
eval1=eval1.dropna(axis=1, how='all')
eval1_len=eval1.size
#Convert column to string and do utf-8 encoding 
eval1['message'] = eval1['message'].astype(str)
eval1.message.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

clean_msg2=[]
#Call to msg_preprocessing method in Haptik.py for data cleaning
for i in range(0,eval1_len):
    clean_msg2.append(phrase_detect.msg_processing(eval1['message'][i]))
eval1['cleaned']=clean_msg2
eval1['cleaned'] = eval1['cleaned'].str.replace('\d+', '')

classification=[]
vectorizer=CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)
vectorizer.fit_transform(X_train)
i=0

for i in range(0,eval1_len):
    
    Y=eval1['cleaned'][i:i+1]
    validation_data=vectorizer.transform(Y)
    validation_data=validation_data.toarray()
    classification.append(loaded_model.predict(validation_data))
eval1["output"]=classification

#print eval1["output"]
#Writing output to out.csv file
print("Writing to final_output.csv")
eval1.to_csv('final_output.csv', sep=',')
print("Completed!!")