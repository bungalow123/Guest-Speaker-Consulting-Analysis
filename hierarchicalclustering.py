import xlrd
import re
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
import pandas as pd
import numpy as numpy
from nltk.stem import WordNetLemmatizer
from pywsd.utils import lemmatize_sentence
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

### data reading + prep
lemmatizer = WordNetLemmatizer()

loc = ("tagged_comments.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

#lemmatize the sentences
temp_comment = []
for i in range(sheet.nrows):
    temp_comment.append(" ".join(lemmatize_sentence(re.sub(r'[^a-zA-Z0-9 ]', '', str(sheet.cell_value(i, 0))))))

temp_speaker = []
for j in range(sheet.nrows):
    temp_speaker.append(str(sheet.cell_value(j, 1)))

temp_tag = []
for k in range(sheet.nrows):
    temp_tag.append(str(sheet.cell_value(k, 2)))

data_tuples = list(zip(temp_comment, temp_speaker, temp_tag))
data = pd.DataFrame(data_tuples, columns = ["comments", "speaker", "tag"])



### SWEETNAM
sweetnam_comments = data[data['speaker'] == "Sweetnam"]

text1 = sweetnam_comments.comments.values
stop_words = ["sweetnam", "sweetnams", "mrsweetnam", "mrsweetnams", "mr", "quinn", "data", "many", "talk", "learn", "hi"] + list(STOPWORDS)

### ANDERSON
anderson_comments = data[data['speaker'] == "Anderson"]

text2 = anderson_comments.comments.values
stop_words2 = ["anderson", "andersons", "dranderson", "drandersons", "dr", "data", "many", "talk", "learn"] + list(STOPWORDS)


### KRICORIAN
kricorian_comments = data[data['speaker'] == "Kricorian"]

text3 = kricorian_comments.comments.values
stop_words3 = ["kricorian", "kricorians", "drkricorian", "drkricorians", "dr", "data", "many", "talk", "learn"] + list(STOPWORDS)

#This function extracts the key words used in a sentence by exluding stopwords
def word_extraction(sentence,stopwords):    
    ignore = stopwords 
    words = re.sub("[^\w]", " ",  sentence).split()    
    cleaned_text = [w.lower() for w in words if w not in ignore]    
    return cleaned_text

#This function extracts key vocabulary words in a sentence, ignore stopwords and repeated vocab
def tokenize(comments, stopwords):    
    words = []    
    for comment in comments:    
        #print(comment)
        w = word_extraction(comment, stopwords) 
        #print("w is", w)
        words.extend(w)            
    #print(words)
    words = sorted(list(set(words)))
    return words


###SWEETNAM

#Create an array, where columns are the vocab words and each row shows the word count of each vocab word in the comment.
allsweetnam = sweetnam_comments.comments.values
vocab = tokenize(allsweetnam, stop_words)
word_freq = [] 
for sweetnam in allsweetnam:       
    words = word_extraction(sweetnam, stop_words)     
    bag_vector = numpy.zeros(len(vocab))      
    for w in words:         
        for i,word in enumerate(vocab):                
            if word == w:                     
                bag_vector[i] += 1                       
    word_freq.append(numpy.array(bag_vector))

#Make it into a data frame, then normalize it to find the total probability of each vocab word appearing in each comment
df = pd.DataFrame(data=word_freq)
df.columns = vocab
data1 = normalize(df)
data_scaled = pd.DataFrame(data1, columns=df.columns)

#Create a Dendrogram to analyze the number of optimal clusters
plt.figure(figsize=(10, 8))  
plt.title("Sweetnam Dendrogram")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.axhline(y=1.75, color='r', linestyle='--') #4 clusters
plt.show()

#Cluster the comments, in this case 4 clusters
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
y_cluster = cluster.fit_predict(data_scaled) #cluster the comments into 4 different clusters
print(y_cluster)

#find the index of the comments of each cluster
tf_data0 = y_cluster==0
indx0 = [i for i, x in enumerate(tf_data0) if x]
print("indx0", indx0)

tf_data1 = y_cluster==1
indx1 = [i for i, x in enumerate(tf_data1) if x]
print("indx1", indx1)

tf_data2 = y_cluster==2
indx2 = [i for i, x in enumerate(tf_data2) if x]
print("indx2", indx2)

tf_data3 = y_cluster==3
indx3 = [i for i, x in enumerate(tf_data3) if x]
print("indx3", indx3)


###ANDERSON
#Create an array, where columns are the vocab words and each row shows the word count of each vocab word in the comment.
allanderson = anderson_comments.comments.values
word_freq2 = [] 
vocab = tokenize(allanderson, stop_words2)
for anderson in allanderson:       
    words = word_extraction(anderson, stop_words2)        
    bag_vector = numpy.zeros(len(vocab))      
    for w in words:         
        for i,word in enumerate(vocab):                
            if word == w:                     
                bag_vector[i] += 1                       
    word_freq2.append(numpy.array(bag_vector))

#Make it into a data frame, then normalize it to find the total probability of each vocab word appearing in each comment
df2 = pd.DataFrame(data=word_freq2)
df2.columns = vocab
data2 = normalize(df2)
data_scaled2 = pd.DataFrame(data2, columns=df2.columns)

#Create a Dendrogram to analyze the number of optimal clusters
plt.figure(figsize=(10, 8))  
plt.title("Anderson Dendrograms")  
dend2 = shc.dendrogram(shc.linkage(data_scaled2, method='ward'))
plt.axhline(y=2.25, color='r', linestyle='--') #2 clusters
plt.show()

#Cluster the comments, in this case 2 clusters
cluster2 = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
y_cluster2 = cluster2.fit_predict(data_scaled2) #cluster the comments into 2 different clusters
print(y_cluster2)

#find the index of the comments of each cluster
a_tf_data0 = y_cluster2==0
a_indx0 = [i for i, x in enumerate(a_tf_data0) if x]
print("indx0", a_indx0)

a_tf_data1 = y_cluster2==1
a_indx1 = [i for i, x in enumerate(a_tf_data1) if x]
print("indx1", a_indx1)


###KRICORIAN
#Create an array, where columns are the vocab words and each row shows the word count of each vocab word in the comment.
allkricorian = kricorian_comments.comments.values
word_freq3 = []
vocab = tokenize(allkricorian, stop_words3) 
for kricorian in allkricorian:       
    words = word_extraction(kricorian, stop_words3)       
    bag_vector = numpy.zeros(len(vocab))      
    for w in words:         
        for i,word in enumerate(vocab):                
            if word == w:                     
                bag_vector[i] += 1                       
    word_freq3.append(numpy.array(bag_vector))

#Make it into a data frame, then normalize it to find the total probability of each vocab word appearing in each comment
df3 = pd.DataFrame(data=word_freq3)
df3.columns = vocab
data3 = normalize(df3)
data_scaled3 = pd.DataFrame(data3, columns=df3.columns)

#Create a Dendrogram to analyze the number of optimal clusters
plt.figure(figsize=(10, 8))  
plt.title("Kricorian Dendrogram")  
dend3 = shc.dendrogram(shc.linkage(data_scaled3, method='ward'))
plt.axhline(y=1.75, color='r', linestyle='--') 
plt.show()

#Cluster the comments, in this case 4 clusters
cluster3 = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
y_cluster3 = cluster3.fit_predict(data_scaled3) #cluster the comments into 4 different clusters
print(y_cluster3)

#find the index of the comments of each cluster
k_tf_data0 = y_cluster3==0
k_indx0 = [i for i, x in enumerate(k_tf_data0) if x]
print("indx0", k_indx0)

k_tf_data1 = y_cluster3==1
k_indx1 = [i for i, x in enumerate(k_tf_data1) if x]
print("indx1", k_indx1)

k_tf_data2 = y_cluster3==2
k_indx2 = [i for i, x in enumerate(k_tf_data2) if x]
print("indx2", k_indx2)

k_tf_data3 = y_cluster3==3
k_indx3 = [i for i, x in enumerate(k_tf_data3) if x]
print("indx3", k_indx3)
