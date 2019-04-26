from gensim.models.keyedvectors import KeyedVectors
from DocSim import DocSim
from datetime import datetime
import pandas as pd
import string

googlenews_model_path = './data/GoogleNews-vectors-negative300.bin'
stopwords_path = "./data/stopwords_en.txt"

model = KeyedVectors.load_word2vec_format(googlenews_model_path, binary=True)
target_docs=[]
target_docs2=pd.read_excel("DumpComments.xlsx")
target_docs2.dropna(inplace=True)
print(target_docs2.head())
target_docs1=target_docs2["Comments"].tolist()
feature_questions_lwr = [x.lower() for x in target_docs1]
for i in feature_questions_lwr:
    translation_table = dict.fromkeys(map(ord, string.punctuation), ' ')
    string2 = i.translate(translation_table)  # translating string1
    target_docs.append(string2)
with open(stopwords_path, 'r') as fh:
    stopwords = fh.read().split(",")
ds = DocSim(model,stopwords=stopwords)

# source_doc = "how to delete an invoice"
source_doc='''Your customer wanted to lower down the bill and asked about package information. You failed to ask question to provide better package. This would have made the customer feel less worried. You missed identifying household and lifestyle needs of the customer&amp;#44; you could have asked about usage of the current services for better promotions. Probing for service would have helped you in identifying services tailored to customer&apos;s needs. You know Comcast products really well&amp;#44; so be sure to ask more probing questions about how they are using their services which may open the door for you to share even more specifics about packages that would be helpful to the customer.\n
'''
# target_docs = ['delete a invoice', 'how do i remove an invoice', "purge an invoice"]
print(target_docs)
start=datetime.now()
sim_scores = ds.calculate_similarity(source_doc, target_docs)
print (datetime.now()-start)
print(sim_scores)

# Prints:
#   [ {'score': 0.99999994, 'doc': 'delete a invoice'},
#   {'score': 0.79869318, 'doc': 'how do i remove an invoice'},
#   {'score': 0.71488398, 'doc': 'purge an invoice'} ]
