import xlrd
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import WordNetLemmatizer


# lemmatizer from NLTK's package
lemmatizer = WordNetLemmatizer()



### data reading + cleaning
loc = ("~/Desktop/stats 141sl/tagged_comments.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

## remove special characters from comments and lemmatize before appending to a list
temp_comment = []
for i in range(sheet.nrows):
    temp_comment.append(" ".join(lemmatize_sentence(re.sub(r'[^a-zA-Z0-9 ]', '', str(sheet.cell_value(i, 0))))))

## temporary list of speakers
temp_speaker = []
for j in range(sheet.nrows):
    temp_speaker.append(str(sheet.cell_value(j, 1)))

## temporary list of tag values
temp_tag = []
for k in range(sheet.nrows):
    temp_tag.append(str(sheet.cell_value(k, 2)))

## zip lists
data_tuples = list(zip(temp_comment, temp_speaker, temp_tag))
## create pandas DataFrame (to make data easier to work with)
data = pd.DataFrame(data_tuples, columns = ["comments", "speaker", "tag"])




### SWEETNAM
sweetnam_comments = data[data['speaker'] == "Sweetnam"]

text = sweetnam_comments.comments.values # separates sentences into its individual text values
## extra stopwords in addition to the ones in STOPWORD as provided by the 'wordcloud' package
stop_words = ["sweetnam", "sweetnams", "mrsweetnam", "mrsweetnams", "mr", "quinn", "data", "many", "talk", "learn", "hi", "know", "important", "use", "different", "thing", "make", "good", "need"] + list(STOPWORDS)
wordcloud = WordCloud(
    width = 800,
    height = 800,
    background_color = 'black',
    collocations = False,
    stopwords = stop_words).generate(str(text))
fig = plt.figure(
    figsize = (8, 8),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad = 0)


### ANDERSON
anderson_comments = data[data['speaker'] == "Anderson"]

text2 = anderson_comments.comments.values
stop_words2 = ["anderson", "andersons", "dranderson", "drandersons", "dr", "data", "many", "talk", "learn", "know", "important", "use", "different", "thing", "make", "good", "need"] + list(STOPWORDS)
wordcloud2 = WordCloud(
    width = 800,
    height = 800,
    background_color = 'black',
    collocations = False,
    stopwords = stop_words2).generate(str(text2))
fig = plt.figure(
    figsize = (8, 8),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud2, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad = 0)


### KRICORIAN
kricorian_comments = data[data['speaker'] == "Kricorian"]

text3 = kricorian_comments.comments.values
stop_words3 = ["kricorian", "kricorians", "drkricorian", "drkricorians", "dr", "data", "many", "talk", "learn", "know", "important", "use", "different", "thing", "make", "good", "need"] + list(STOPWORDS)
wordcloud3 = WordCloud(
    width = 800,
    height = 800,
    background_color = 'black',
    collocations = False,
    stopwords = stop_words3).generate(str(text3))
fig = plt.figure(
    figsize = (8, 8),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud3, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad = 0)

## displays all three word clouds
plt.show()
