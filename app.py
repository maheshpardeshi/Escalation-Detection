from flask import Flask,request
from flask_restful import Resource, Api
from pycorenlp import StanfordCoreNLP
from nltk.corpus import stopwords   
import nltk
import pandas as pd
import spacy 
import simplejson as json
from collections import defaultdict
from datetime import date
import numpy as np
from datetime import datetime, timedelta
import re
from nltk.tokenize import word_tokenize
import pickle
from nltk.stem import WordNetLemmatizer 
from html.parser import HTMLParser
import re
from nltk import RegexpParser
from nltk.tree import Tree
from applicationinsights.flask.ext import AppInsights
import traceback
import sys
from applicationinsights import TelemetryClient
import logging
import tensorflow as tf
from applicationinsights import channel
from applicationinsights.logging import LoggingHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import keras.backend.tensorflow_backend as tb
#Code to hanlde the exceptions and logs to App Insights in Azure
tc = TelemetryClient('514ffbdf-cab4-4207-85fc-f3eb5c270d54')
# set up channel with context
telemetry_channel = channel.TelemetryChannel()
telemetry_channel.context.application.ver = '1.0.0'
telemetry_channel.context.properties['Intent'] = 'Escalation'

#Set up logging
handler = LoggingHandler('514ffbdf-cab4-4207-85fc-f3eb5c270d54', telemetry_channel=telemetry_channel)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger = logging.getLogger('simple_logger')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

#Log something (this will be sent to the Application Insights service as a trace)
app = Flask(__name__)
appinsights = AppInsights(app)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
nnps=["Dear All","Dear Team","Dears","Hi All","Hi Team","Hi","Hey","Hello All","Hello Team","Hi guys"]
NegativeSents=[]

global graph
graph = tf.compat.v1.get_default_graph()

#Standford Core NLP Azure dockers URL to check the sentiment of the text
nlps = StanfordCoreNLP('https://sentimentanalysis-dev-api.azurewebsites.net/')
allPwps_Avns=[]
filename = "/var/LSTM_model_EscalationDetection.sav"            
filenameTok="/var/Escalation_tok.sav"
model = pickle.load(open(filename, 'rb'))
tok=pickle.load(open(filenameTok, 'rb'))

filename_Appr = "/var/LSTM_model_AppreciationDetection.sav"            
filenameTok_Appr="/var/tok.sav"

model_Appr = pickle.load(open(filename_Appr, 'rb'))
tok_Appr=pickle.load(open(filenameTok_Appr, 'rb'))

max_words = 15000
max_len = 300

#Load Spacy model to get POS,Dependency matrix,lemma
nlpSpacy = spacy.load('en_core_web_lg')
#pronounsList=['you','me','myself','mine','your','yours','yourself','his','her','herself','he','she','they','them','who','whom','whose']
pronounsList=['you','your','yours','yourself','his','her','herself','he','she','they','them','who','whom','whose','it','i','this']
propNouns=['you','your']
sampleSignatures=['best','regards','cheers','cheer','thanks','sincerely','thank you','thanks and regards','thank you  and  regards','thank you and regards','thanks & regards','yours']
lemmatizer = WordNetLemmatizer() 
work_related_nouns=['effort','review','work','worker','job','performance','perform','task','assignment','attempt','moil','achievement',
'creation','go','energy','accomplishment','creation','success','victory','acquirement','acquisition','initiative',
'action','stuff','hero','milestone','demonstration','workshop','sale','marketing','deal','support','documentation','observation',
'contribution','achieved','achieve','idea','test','coverage','plan','doc','dedication','attitude','documentation','assistence','review',
'achivement','test','testing','test case','test cases','testcase','employee','deployment','solution','developer','design','UI','designing',
'handle','skill','implementation','creativity','delivery','help','supervision','progress','team','leader','colleague','guys','guy','lead',
'quality','coworker','session','person','frontend','team','member','player','coder','knowledge','teamwork','result',
'experience','buddy','approach','people','advice','enthusiasm','boy','project','man','woman','women','development','suggestion','report']

mostlikelyApprAdjectives=['cool','great','awesome','nice','amazing','amazed','best','bright','brilliant','calm','carefull'
                         'charm','clever','confident','congrats','enthusiastic','excellent','expert','fabulous','fantastic'
                         'dignity','good','like','perfect','stromg','wonderfull','dedicate','wise','well','phenomenal',
                         'Kudos','hardwork','consistent','positive','success','intelligent','ambition','creative','passion','efficient'
                         'extraordinary','immense','cooperate','genius','progress','outstanding','pleasure','rockstar','asonishing'
                         ]
ADJ_Score_CSV=pd.read_csv("/var/PossWordsDF.csv",encoding="ISO-8859-1")
ADJ_Score_DF=ADJ_Score_CSV[ADJ_Score_CSV['Sentiment']<1]
ADJ_Score_DF.reset_index(inplace=True, drop=True)

PosetiveADJ_Score_DF=ADJ_Score_CSV[ADJ_Score_CSV['Sentiment']>1]
PosetiveADJ_Score_DF.reset_index(inplace=True, drop=True)

app.config['APPINSIGHTS_INSTRUMENTATIONKEY'] = '514ffbdf-cab4-4207-85fc-f3eb5c270d54'
replace_list = {r"i'm": 'i am',
                r"'re": ' are',
                r"let's": 'let us',                
                r"'ve": ' have',
                r"can't": 'can not',
                r"cannot": 'can not',
                r"shan't": 'shall not',
                r"n't": ' not',
                r"'d": ' would',
                r"'ll": ' will',
                r"'scuse": 'excuse',                
                '!': '.',                
                '\s+': ' '}
# To check is the text is imperative or non imperative text.
def is_imperative(dep):
    tagged_sent=[]
    for k in dep:
        temp=  [k[0],k[5],k[4]]
        tagged_sent.append(tuple(temp))
        
    # if the sentence is not a question...
    if len(tagged_sent)>0 and tagged_sent[-1][0] != "?":
        # catches simple imperatives, e.g. "Open the pod bay doors, HAL!"
        if(tagged_sent[0][2] in ['ADV']):
            return False
        if (tagged_sent[0][1] == "VB" or tagged_sent[0][1] == "VBZ" or tagged_sent[0][1] == "VBP" or 
        tagged_sent[0][1] == "VBD" or tagged_sent[0][1] == "VBN"  or tagged_sent[0][1] == "MD"):
            return True

        # catches imperative sentences starting with words like 'please', 'you',...
        # E.g. "Dave, stop.", "Just take a stress pill and think things over."
        else:
            chunk = get_chunks(tagged_sent)
            # check if the first chunk of the sentence is a VB-Phrase
            if type(chunk[0]) is Tree and chunk[0].label() == "VB-Phrase":
                return True

    # Questions can be imperatives too, let's check if this one is
    else:
        # check if sentence contains the word 'please'
        pls = len([w for w in tagged_sent if w[0].lower() == "please"]) > 0
        # catches requests disguised as questions
        # e.g. "Open the doors, HAL, please?"
        chunk = get_chunks(tagged_sent)
        
        if pls and (tagged_sent[0][1] == "VB"  or tagged_sent[0][1] == "VBZ" or tagged_sent[0][1] == "VBP" or 
        tagged_sent[0][1] == "VBD" or tagged_sent[0][1] == "VBN" or tagged_sent[0][1] == "MD"):
            return True
        
        # catches imperatives ending with a Question tag
        # and starting with a verb in base form, e.g. "Stop it, will you?"
        elif type(chunk[-1]) is Tree and chunk[-1].label() == "Q-Tag":
            if (chunk[0][1] == "VB" or
                (type(chunk[0]) is Tree and chunk[0].label() == "VB-Phrase")):
                return True
            
        elif (tagged_sent[0][1] == "VB"  or tagged_sent[0][1] == "VBZ" or tagged_sent[0][1] == "VBP" or
              tagged_sent[0][1] == "VBD" or tagged_sent[0][1] == "VBN" or tagged_sent[0][1] == "MD"):
            return True
    return False

# chunks the sentence into grammatical phrases based on its POS-tags
def get_chunks(tagged_sent):
    chunkgram = r"""VB-Phrase: {<DT><,>*<VB>}
                    VB-Phrase: {<RB><VB>}
                    VB-Phrase: {<UH><,>*<VB>}
                    VB-Phrase: {<UH><,><VBP>}
                    VB-Phrase: {<PRP><VB>}
                    VB-Phrase: {<NN.?>+<,>*<VB>}
                    Q-Tag: {<,><MD><RB>*<PRP><.>*}"""
    chunkparser = RegexpParser(chunkgram)
    return chunkparser.parse(tagged_sent)

# Find the signature start in the array of sentences.
def getMailSignature(f,e):
    if(f!='' and e!=''):
        if(f.lower().strip().find(e.lower().strip())>=0):        
            return True
        elif(e.lower().strip().find(f.lower().strip())>=0):        
            return True
    return False   

# Detect the sentence is action required sentence or else
def getActionRequiredText(sentence,dep):
    actionRequiredWords=['need','have to','has to','have been','must', 'ought', 'shall', 'should','will','might','if','can','would be']    
    imp=True
    #isimerativeSent=bool([s for s in sentence.split(' ') if s.lower() in actionRequiredWords])
    isimerativeSent=bool([a for a in actionRequiredWords if sentence.strip().lower().find(a)>-1])
    #isimerativeSent=any(word in sentence for word in actionRequiredWords)
    for d in dep:
        if(d[6].lower() in ['expect','hope','hopefully']):
            return True
    if(len(dep)>0 and dep[0][0].lower() in 'thank'):
        return False
    
    if(not isimerativeSent): 
        isimerativeSent=is_imperative(dep)
   
    return isimerativeSent

#app.config('ServerName','13.71.1.136')
def getPositivityUsingSFCoreNLP(sent,dep):
    sent=getPOSRemovedSent(dep)
    try:        
        res = nlps.annotate(sent,properties={'annotators': 'sentiment','outputFormat': 'json','timeout': 1000,})
        for s in res["sentences"]:
            if(s["sentiment"] =='Negative'):
                sent=getPOSRemovedSent(dep,True)
                res = nlps.annotate(sent,properties={'annotators': 'sentiment','outputFormat': 'json','timeout': 1000,})
                for s in res["sentences"]:
                    return(s["sentiment"])                    
            return(s["sentiment"])
    except:
        pass

#Check sentance tartgeted to user is present in list of employees or TO or CC employees
def isKnownUser(empName,TOUsers,CCUsers):           
    ToUsersList=TOUsers.split(";")
    CCUsersList=CCUsers.split(";")    
    for t in ToUsersList:
        if(t.lower().find(empName.lower())>-1):
            return t
    for t in CCUsersList:        
        if(t.lower().find(empName.lower())>-1):
            return t
    return []

#check all words are proper nouns only
def onlyPropNounsInSents(ent):
    onlyPropNounsInSent=True 
    #Filter our word dear from mails starting    
    ent=[e for e in ent if e[0].lower() not in ['dear']]
    #Filter out injectives and determinent, Proper nouns and punct 
    for e in range(len(ent)): 
        if(ent[e][4] not in ['INTJ','DET','PROPN','PUNCT','CCONJ']):
            onlyPropNounsInSent=False
            break
    return onlyPropNounsInSent

#Find the name of targeted user in sentance
def getTargetedUsers(en):        
        users=[]
        for i in range(len(en)):
            if(i!=len(en)-1):       
                pren=en[i-1]
                fi=en[i]
                si=en[i+1]  
                if(fi[4]==si[4]=='PROPN'):
                    users.append(fi[0]+" "+si[0])
                    i=i+1   
                elif(fi[4]=='PROPN' and fi[1]=='compound' and si[4]=='PROPN'):
                    users.append(fi[0]+" "+fi[2])
                    i=i+1   
                elif(fi[4]=="PROPN" and si[4]!="PROPN" and i>=1 and pren[4]!="PROPN" and fi[0] not in users ):
                    users.append(fi[0])
                elif(fi[4]=="PROPN" and si[4]!="PROPN" and i==0 and fi[0] not in users ):
                    users.append(fi[0].replace("@",""))    
            else:   
                fi=en[i]
                pren=en[i-1]                
                if(fi[4]=="PROPN"  and i>=1 and pren[4]!="PROPN"):
                    users.append(fi[0])        
                if(len(en)==1 and fi[4]=="PROPN"):
                    users.append(fi[0])        
        
        return users  

#Check is there any proper noun in sentance
def check_PROPNPresent(ent,Mail_To,Mail_CC): 
    isPresent=[]
    NotPresent=[]        
    users=getTargetedUsers(ent)
    for j in range(len(users)):
        KnownUsers= isKnownUser(users[j],Mail_To,Mail_CC)
        if(len(KnownUsers)>0):
            isPresent.append(KnownUsers)
        else:
            NotPresent.append(users[j])   
    return isPresent,NotPresent    
    
# Find the sentnece is talking about which PROPN (Proper Noun)    
def findSentanceUserMapping(mailBody,mailFrom,mailTo,mailCC):
    Sentance_TargetedUser=[]    
    tf=""    
    tfu=""
    updatedSentAfterYou=""
    # Iterate over list of sentences in text. tf:targetedFor,tfu:targetedForIfYou
    for m in mailBody:        
        d = {}         
        if(m.strip()!='' ):  
            m=m.replace('@','').replace('&',' and ')                        
            dep=getDependency(m,"Escalation")            
            onlyPropNounsInSent=onlyPropNounsInSents(dep)
            isPresent,NotPresent=check_PROPNPresent(dep,mailTo,mailCC)
            #Find the proper noun and sentence mapping
            if(len(isPresent)>0 ):                
                tf=";".join([t for t in isPresent])                
                if(onlyPropNounsInSent):                     
                    tfu=tf
            if(len(isPresent) is 0):
                if(len(tf)==0):
                    tf= "".join([t for t in mailTo])  

            if(any(p in ['you','yours','yourself'] for p in m.lower())):
                if(tfu!=''): 
                    tf=tfu
                updatedSentAfterYou=m.replace('you',tf)
                updatedSentAfterYou=m.replace('You',tf)
            #if(' i ' in m.lower() and checkPronounInSent(m.lower())):
            #    tf=mailFrom
            d[tf] = m
            Sentance_TargetedUser.append(json.dumps(d))            
    return Sentance_TargetedUser

# Check the pronouns in the sentance
def checkPronounInSent(sent):
    return bool([s for s in sent.split(' ') if s.strip().lower() in pronounsList])
  

#Tokanise the sentance
def nltkToknisation(docs):        
    doc = [w[0] for w in docs.values]
    _docs = [[w.lower() for w in word_tokenize(text)] for text in doc]
    return _docs

#Find the dependency in the sentance
def getDependency(sent,model):        
    doc = nlpSpacy(sent)
    dep=[]
    adjs=[]
    if model=="Appreciation":
        adjs=ADJ_Score_DF['Text'].values
    else:
        adjs=PosetiveADJ_Score_DF['Text'].values
    
    for token in doc:
        if(token.text.lower()  in adjs):            
            dep.append([token.text, token.dep_, token.head.text, token.head.pos_,'ADJ',token.tag_,token.lemma_.lower()])
        else:
            dep.append([token.text, token.dep_, token.head.text, token.head.pos_,token.pos_,token.tag_,token.lemma_])
    
    return dep
def filterDataType(sent):
    types=['TIME','DATE']
    doc = nlpSpacy(sent)
    entities=[(i, i.label_, i.label) for i in doc.ents]
    return not bool([e for e in entities if e[1] in ['TIME']])
# Find the subject of the sentance
def findSubjectInSent(dep):    
    allSubs=[]
    for i in range(len(dep)):
        try:
            if(dep[i][1]=='nsubj'):
                 allSubs.append([dep[i][0],dep[i][4]])  
            if(dep[i][1]=='poss' and dep[i][4]=='ADJ'):
                allSubs.append([dep[i][0]])
        except:        
            pass
    return allSubs

# Get Part Of speach from the sentance
def getPOSTag(dep,pos):
    try:
        if(pos in ['NOUN','VERB','PROPN']):  
            return [x for x in dep if x[4] == pos][0][0]    
        if(pos in ['ADJ']):        
            return [x for x in dep if x[4] == pos and x[1]!='poss'][0][0]
        if(pos in ['amod']):        
            amod=[x for x in dep if x[1] == pos][0]    
            return [amod[0][0],amod[0][2]]
        if(pos in ['aux']):        
            return [x for x in dep if x[1] == pos or x[4] ==pos][0][0]    
        if(pos in ['PRON']):        
            return [x for x in dep if len(x)>0 and(x[1] == pos or x[0].lower() =='your')][0][0]
        if(pos in ['VBG']):        
            return [x for x in dep if x[5] == pos][0][0]    
    except:
        pass

# Find the tesne of the sentances
def getTenseOfSent(dep):
    tense = {}
    past=fut=pres=0   
    
    for i in range(len(dep)):         
        if(dep[i][5] in ["MD","VBC","VBF"]):
            fut=fut+1
        if(dep[i][5] in ["VBP", "VBZ","VBG","VB"]):
            pres=pres+1        
        if(dep[i][5] in ["VBD", "VBN"]):
            past=past+1
        
    tense["future"] =fut
    tense["present"]=pres
    tense["past"] =past    
    return json.dumps(tense)

#Find the object in the element
def find(l, elem):
    for row, i in enumerate(l):
        try:
            column = i.index(elem)
        except ValueError:
            continue
        return row, column
    return -1

#Return the ith row from matrix
def column(matrix, i):
    return [row[i] for row in matrix]

#To check is the sentence is not in future tense or requested sentence
def isValiedText(sent,dep,mail_To,mail_CC,pwps):
    tense=json.loads(getTenseOfSent(dep))   
    _aux=0
    aux=getPOSTag(dep,'aux')    
    if(aux!='' and aux is not None):
        _aux=len(aux)
    vbg=getPOSTag(dep,'VBG')    
    _vbg=0    
    
    #Get the gerund verb
    if(vbg!='' and vbg is not None):
        _vbg=len(vbg)
    
    #Find out is any proper nouns from TO or CC present in text    
    presentUser=check_PROPNPresent(dep,mail_To,mail_CC)[0]
    
    #Get the text in requested form
    if(any('please' in x.lower() for x in nltk.word_tokenize(sent))):
        return False,presentUser
    
    #Returns false if text is in future tense 
    if(tense['future']>0 and tense['past']==0 and tense['present']>0 and _aux>0 and _vbg==0):    
        return False,presentUser
    
    if(len(pwps)==0 and isAppreciateKeywordPresent(dep)):                
        return False,presentUser
    return True,presentUser

#Get POS removed sentance for check sentance polarity
def getPOSRemovedSent(dep,secondOpenion=False):
    updatedSent=''    
    removePOS=['PROPN','INTJ']
    auxWords=['was','had','has','have','will','would','should']
    IntjWords=['hi','dear','hello','hey','ah','oh','hmm','ouch','uh','just','really','everyone','too','quickly']
    for i in range(len(dep)):        
        if(dep[i][1] not in removePOS and dep[i][4] not in removePOS and dep[i][0].lower() not in IntjWords):         
            updatedSent=updatedSent+' '+dep[i][0]
        if(secondOpenion and dep[i][0].lower() not in auxWords):
            updatedSent=updatedSent+' '+dep[i][0]
            
    updatedSent=updatedSent.replace("n't",'not').replace("Well","well")
    return updatedSent

#Is positive words from the list of positive words present
def isPossitiveWordPresent(dep,model):          
    possWords=[]    
    adjs=[]
    if model=="Appreciation":        
        for i in range(len(dep)):
            lemma=dep[i][6]        
            if(PosetiveADJ_Score_DF[PosetiveADJ_Score_DF['Text']==lemma].shape[0]>0):         
                possWords.append([dep[i][0],PosetiveADJ_Score_DF[PosetiveADJ_Score_DF['Text']==lemma][['Text','Sentiment']].values[0][1]])
    else:                
        for i in range(len(dep)):
            lemma=dep[i][6]        
            if(lemma in ADJ_Score_CSV.Text.values):         
                possWords.append([dep[i][0],ADJ_Score_CSV[ADJ_Score_CSV.Text ==lemma].Sentiment.values[0]])

    return possWords

#Find out is given text contain any commanly used adjectives 
def isCommanPositiveWordPresent(dep,model):          
    possWords=[]
    try:
        if model=="Appreciation":    
            for i in range(len(dep)):
                lemma=dep[i][6]       
                if(PosetiveADJ_Score_DF[PosetiveADJ_Score_DF['Text']==lemma].shape[0]>0):
                    possWords.append([dep[i][0],PosetiveADJ_Score_DF[PosetiveADJ_Score_DF['Text']==lemma][['Text','Sentiment']].values[0][1]])
        else:
            for i in range(len(dep)):
                lemma=dep[i][6]       
                if(lemma in ADJ_Score_CSV.Text.values):         
                    possWords.append([dep[i][0],ADJ_Score_CSV[ADJ_Score_CSV.Text ==lemma].Sentiment.values[0]])

        return possWords
    except:
        return possWords

#check is defined Escalation verbs peresent
def isApprVerbNounPresent(dep):
    appWords=[]
    for i in range(len(dep)):
        if(dep[i][6].lower() in work_related_nouns and (dep[i][4] in ['NOUN','PROPN','VERB','ADJ'] or dep[i][5] in ['VBG','NN']) ):
            appWords.append(dep[i][0].lower())
    return appWords

def isAppreciateKeywordPresent(dep):
    return bool([d for d in dep if d[6] in ['appreciate'] and d[1] not in ['compound']])

#get the simple seneances in array from the complex sentance
def seperateSent(sent,dep):    
    independentSent=[]
    Sentences=getIndependentSent(dep)
    if(Sentences==[]):
        Sentences.append(sent)
    return Sentences

#split the sentence by injector and punctuate 
def getIndependentSent(dep):
    independentSent=[]    
    newStart=False
    words=[]
    compoundSentences=[]
    start=0
    for i in range(len(dep)):
        #Find the text with more than one subjects
        if(column(dep,1).count('nsubj')>1):  
            if(i==len(dep)-1 ):            
                words.append(dep[i])
                if(words!=[]):
                    independentSent.append(words)  
            #Seperate the sentence by punctuate and conjuction
            if(((dep[i][1] == 'punct' and i>0 and dep[i-1][4] in ['PROPN']) or (dep[i][1] =='cc' and dep[i][0] not in ['and','or']) or dep[i][1]== 'mark') and i!=len(dep)-1 ):
                newStart=True
            if(newStart): 
                if(words!=[]):                    
                    independentSent.append(words)
                words=[]
                newStart=False
            if(newStart == False and dep[i][1] != 'punct' and dep[i][1] != 'mark' and i!=len(dep)-1 ):      
                words.append(dep[i])            
        else:          
            if(column(dep,1).count('cc')>0 and column(dep,1).count('conj') >0):
                if(dep[i][1] =='cc' and dep[i][0] not in ['and','or']):
                    cSents=[]
                    for k in range(start,i):                                                
                        cSents.append(dep[k])                        
                    start=i
                    if(cSents!=[]):                        
                        compoundSentences.append(cSents)
                if(i==len(dep)-1):
                    cSents=[]                    
                    for k in range(start,i+1):                                                  
                        cSents.append(dep[k])
                    start=i
                    if(cSents!=[]):
                        compoundSentences.append(cSents)                        
    if(len(words)>0):        
        compoundSentences=independentSent
    
    if(len(compoundSentences)>0):
        i=-1
        while(i!=len(compoundSentences)):
            i=i+1       
            try:
                conjPresent=False  
                for j in range(len(compoundSentences[i])):            
                    if(compoundSentences[i][j][1]=='conj' 
                       and (compoundSentences[i][j][3]=='PROPN'
                       or compoundSentences[i][j][4]=='PROPN'
                       or compoundSentences[i][j][4]=='PRON'  )):                                                      
                        for k in range(len(compoundSentences[i])):                    
                            compoundSentences[i-1].append(compoundSentences[i][k])
                        del compoundSentences[i]
                
                if(i !=0 and column(compoundSentences[i],1).count('conj')>0):                    
                    conjPresent=True

                if(i !=0 and conjPresent==False):
                    for k in range(len(compoundSentences[i])):                    
                        compoundSentences[i-1].append(compoundSentences[i][k])
                    del compoundSentences[i]
                    i=i-1
            except:                
                 pass
        sepSent=[]
        #Join the seperate sentences if conjunct present
        for i in range(len(compoundSentences)):
                if(compoundSentences[i]!=[]): 
                    txt=''    
                    for j in range(len(compoundSentences[i])):
                        if(len(compoundSentences[i][j])>0 ):                             
                            if((j==0 and compoundSentences[i][j][1] in ['cc'])):
                                txt=txt+''
                            else:
                                txt=txt+' '+compoundSentences[i][j][0]                               
                            
                sepSent.append(txt)         
   
    if(len(compoundSentences)>0):
        independentSent=sepSent
    return independentSent

#Detect the object about which we are talking in the text.
def getObjectInSent(dep_in,presentUsers):    
    entityNames=[]
    if(len(dep_in)>1 and len(dep_in[1])>0):
        for d in dep_in[1]:
            if((d[4] in ['PROPN'] and d[4] not in presentUsers)):                
                entityNames.append(d[0])
    return entityNames

#Find the sentences with thanks word
def isThanksPresent(sent):
    if(sent.lower().strip().find('thank you')>=0): 
        return True
    return False
    
    
# Get complex sentence sepeated by inject
def seperateSentenceByIN(dep):
    inSeperatedSents=[]
    inFirstPart=[]
    inSecondPart=[]
    inFound=False
    for d in dep:        
        if d[5] =="IN":                
            inFound=True
        if(not inFound):            
            inFirstPart.append(d)
        else:
            inSecondPart.append(d)
    inSeperatedSents.append(inFirstPart)   
    inSeperatedSents.append(inSecondPart)
    return inSeperatedSents

# Check adjective modifier present in sentence
def isAdjModExist(dep,pwps):    
    for i in range(len(dep)):        
        if(i<len(dep)-1 and dep[i][1] in ['amod','advmod'] and dep[i+1][0].lower() in pwps):            
            return True
    return False

#Check is the adjective noun present
def OnlyAdjNounPairPresent(dep,model):    
    dep=list(filter(lambda x : x[1] not in ['aux','det'], dep))
    verb=column(dep,4).count('VERB')
    noun=column(dep,4).count('NOUN')
    adj=column(dep,4).count('ADJ')    
    pron=[d[0] for d in dep if d[4]=='PRON' and d[0].lower() in propNouns]
    df=pd.DataFrame(dep)
    if(df.shape[0]>0):    
        dep=[d for d in dep if d[4] not in ['PUNCT','CCONJ','PROPN']]
    if(len(pron)>0):
        dep=[d for d in dep if d[4] != 'PRON']
    aprWord=False
    if(len(dep)<4 and len(dep)>0):   
        if(isCommanPositiveWordPresent(dep,model) and adj>0):    
            if(len(dep)>1 and bool([d for d in dep if d[1] in ['compound']])):
                return False
            if(len(dep)==1):
                aprWord=True
            if(len(dep)==2 and (isApprVerbNounPresent(dep) or(verb==1 and adj==1))):
                aprWord=True        
    return aprWord

#Add 0.5 score if more than one postive adjective present
def adjustMultipleAdjectives(dep,escalationScore):    
    for i in range(len(dep)):
        if(i<len(dep)-1 and dep[i][4] == 'ADJ' and dep[i+1][4] =='ADJ'):            
            escalationScore+=0.5
    return escalationScore    

#Find link in Preposition and adjective    
def isPrepAdjPresent(dep,pwps):
    pobjLst=[d for d in dep if(d[1] is 'pobj')]
    adjLst=[d for d in dep if(d[4] is 'ADJ' and d[0].lower() in pwps)]
    prepLst=[d for d in dep if(d[1] is 'prep')]
    prepAdj=[p for p in prepLst if (p[2] in [p[0] for p in adjLst])]
    objPrep=[o for o in pobjLst if o[2] in [p[0] for p in prepAdj]]
    return bool([o for o in objPrep if o[0].lower() in pwps])

#Find the dependency in verb and adjective e.g. it was PURE AWESOME
def isVerbAppreciated(dep,pwps):    
    verbLst=[d for d in dep if(d[4] is 'VERB')]
    adjLst=[d for d in dep if(d[4] is 'ADJ' and d[0].lower() in pwps)]
    subjLst=[d for d in dep if(d[1] is 'nsubj' and (d[0].lower() in pronounsList) or d[4] in ['PROPN'])]
    verbAdj=[v for v in verbLst if ((v[0] in [a[2] for a in adjLst if a[1] in ['advmod','dobj','acomp']]))]
    subjVerbList=[s for s in subjLst if(s[0].lower() in propNouns and s[2] in [v[0] for v in verbAdj])]    
    if(len(subjVerbList)>0 and len(verbAdj)>0):
        return True
    return False

#Adjective should not be compound word.
def isAdjCompoundWord(dep,pwps):
    if(len([d for d in dep if d[4] =='ADJ' and d[0].lower() in pwps and d[1] in ['compound']])==len(pwps)):
        return True
    return False

def pobj_verb_adj(dep,pwps,avn):
    verbLst=[d[0] for d in dep if(d[1] in ['ROOT','ccomp'] and d[4] in ['ADJ'] and d[6] in pwps)]
    prepLst=[d[0] for d in dep if(d[1] is 'prep' and d[2] in verbLst)]
    return bool([d for d in dep if d[0] in avn and d[1] in ['dobj','pobj'] and (d[2] in verbLst or d[2] in prepLst)])

#You done your every job with full of passion;Well done job
def avn_dobj_adj_verb(dep,pwps,avn):
    rootItem=[r[0] for r in dep if r[1]=='ROOT' and r[4] in ['VERB']]
    subjLst=[d[2] for d in dep if(d[1] is 'nsubj' and (d[0].lower() in pronounsList or d[4] in ['PROPN']) and d[2] in rootItem)]
    avn_dobj=[d for d in dep if d[1] in d[1] in ['dobj'] and d[2] in subjLst and d[0] in avn]
    if(len(avn_dobj)==0):
        avn_dobj=[d for d in dep if d[1] in d[1] in ['dobj'] and d[0] in avn]    
    adjLst=[d for d in dep if(d[4] is 'ADJ' and d[0].lower() in pwps)]
    if(len(adjLst)>0 and len(avn_dobj)>0):
        return True
    return False

#Extract link in adjective and preposition.
def verRoot_adjPobjOfPrep(dep,pwps):
    try:
        rootItem=[r for r in dep if r[1]=='ROOT' and r[4] in ['VERB']]
        subjLst=[d for d in dep if(d[1] is 'nsubj' and d[2] ==rootItem[0][0] and (d[0].lower() in pronounsList or d[4] in ['PROPN']))]
        adjLst=[d for d in dep if(d[4] is 'ADJ' and d[1] is 'pobj')]
        prepLst=[d for d in dep if(d[1] is 'prep')]
        if(len(subjLst)>0 and len(prepLst)>0 and adjLst[0][2] == prepLst[0][0] and adjLst[0][0].lower() in pwps):
            return True
        return False
    except:
        return False

#Get all the adjective and adjective complements and relations in them     
def adj_acomp_root(dep,pwps,avn):
    rootItem=[r for r in dep if r[1]=='ROOT' and r[4] in ['VERB']]    
    adj_Acomp_Root=False  
    if(len(rootItem)>0):
        subjLst=[d for d in dep if(d[1] is 'nsubj' and d[2] ==rootItem[0][0] and (d[0].lower() in avn or d[0].lower() in pronounsList or d[4] in ['PROPN']))]
        if(len(subjLst)>0):
            adjLst=[d for d in dep if(d[4] is 'ADJ' and d[0].lower() in pwps)]
            adj_Acomp_Root=[a for a in adjLst if a[1] in ['acomp','advmod'] and a[2].lower() in rootItem[0][0]]
            if(adj_Acomp_Root==[] and len(adjLst)>0):                   
                conjAdj= [d for d in dep if d[0] in [a[2] for a in adjLst if a[1] in ['conj']]]             
                adj_Acomp_Root = [a for a in conjAdj if (a[1] in ['acomp','advmod'] and a[2].lower() in rootItem[0][0])]            
    return bool(adj_Acomp_Root)

#Find link in adjectvie and verb
def attrAdj_verb(dep,pwps):
    try:
        verbLst=[d[0] for d in dep if(d[4] in ['VERB'] and d[1] in ['ROOT'])]
        adjLst=[d for d in dep if(d[4] is 'ADJ' and d[1] in ['attr'] and d[2] in verbLst and d[0].lower() in pwps)]
        subjLst=[d[0] for d in dep if(d[1] is 'nsubj' and d[2] in verbLst and (d[0].lower() in pronounsList or d[4] in ['PROPN']))]
        if(len(subjLst)>0 and len(adjLst)>0):
            return True
    except:
        return False

#Find out adjective modifiers relation present.
def amod_adj(dep,pwps,avn):
    try:
        subLst=[d[0] for d in dep if d[1] in ['nsubj','ROOT']]        
        amod_adjLst=[d for d in dep if d[4]=='ADJ' and d[1] in ['amod'] and d[2].lower() in avn and d[0].lower() in pwps]
        return bool(amod_adjLst)
    except:
        return False
    
#Extract adjective conjuction     
def conjAdjList(dep,pwps):
    adjLst=[d for d in dep if(d[4] is 'ADJ' and d[1] in ['pobj','conj','acomp'] and d[0].lower() in pwps)]
    rootItem=[r for r in dep if r[1]=='ROOT' and r[4] in ['VERB']]
    conjItem=[a[2] for a in adjLst if a[1] =='conj']
    subjLst=[d[0] for d in dep if(d[1] is 'nsubj' and (d[0].lower() in pronounsList or d[4] in ['PROPN']))]
    conjRow=[]
    if(conjItem!=[]):
        conjRow=[d for d in dep if d[0]==conjItem[0]]
    else:
        conjRow=adjLst
    if(len(conjRow)>0 and len(rootItem)>0 and conjRow[0][2] == rootItem[0][0] and len(subjLst)>0):
        return True
    return False

#Extract adjective objet relationship e.g. She showed a high level of technical skills
def avn_pobj_adj_amod(dep,pwps,avn):
    rootItem=[r for r in dep if r[1]=='ROOT' and r[4] in ['VERB']]
    adjLst=[d for d in dep if(d[4] is 'ADJ' and d[1] in ['pobj','amod'] and d[0] in pwps)]
    prepLst=[d[0] for d in dep if(d[1] is 'prep')]
    pobjLst=[d for d in dep if d[0].lower() in avn and d[1] in ['pobj'] and d[2] in prepLst]
    if(len(pobjLst)>0 and len(adjLst)>0 ):
        return True
    return False

#Extract adjective and noun relationship
def attr_adj_noun(dep,pwps,avn):
    try:
        adjDep=[x for x in dep if x[0].lower() in pwps and x[1] == 'amod' and x[2].lower() in avn]      
        attrLst=bool([d for d in dep if(d[0] in [a[2] for a in adjDep] and d[1] in ['attr'])])
        if(len(attrLst)>0):
            return True
        else:
            return False
    except:
        return False
        
#Logic to get is the text has dependency matrix which proves that text is appreciated text or not. also cross check against LSTM to get extream appreciated texts and neutral text         
def checkEscalationText(Appreciated_User,MailTo,MailCC,Mail_From,Sentence,MailBody):
    dep=getDependency(Sentence,"Escalation")  
    isAdjExist=column(dep,4).count('ADJ')   
    #Check is the Escalation keywords present in the text
    isApprKeyPresent=isAppreciateKeywordPresent(dep)
    #Reject the sentences without any adjectives
    if(isAdjExist>0):  
        #Get the seperate simple sentences from the complex text.
        sents=seperateSent(Sentence,dep)     
        escalationDF=[]
        for j in range(len(sents)):
            escalationScore=0
            isItEscalationText=False
            sent=sents[j].strip()           
            if(sent!='' and filterDataType(sent)):  
                sent=re.sub(r"[^a-zA-Z.,-]+", ' ', sent).strip()
                dep=getDependency(sent,"Escalation")    
                #Get the imperative sentences           
                isImperativeSent=getActionRequiredText(sent,dep)
                onlyPropNounsInSent=onlyPropNounsInSents(dep)  
                pwp=isPossitiveWordPresent(dep,"Escalation")
                pwp=[p for p in pwp if p[1]<0]
                #Do not Process further if isImperative is true or sentence only contains the proper nouns             
                if(len(pwp)>0 and (not isImperativeSent and not onlyPropNounsInSent)):                    
                    adj=column(dep,4).count('ADJ')
                    if(adj>1):
                        escalationScore=adjustMultipleAdjectives(dep,escalationScore)*-1                                            
                    avn=isApprVerbNounPresent(dep)
                    #print("escalationScore",escalationScore)
                    pwps=[]
                    pwps=[p[0].lower() for p in pwp]                   
                    if(len(pwps)>0 and set(avn)==set(pwps)):
                        avn=[]
                        avn_pwpNotSame=False
                        
                    if(len(pwp)>0):
                        escalationScore+=pd.DataFrame(pwp).sort_values(1,ascending=False)[1].max()*-1
                    akp=isAppreciateKeywordPresent(dep)
                    
                    pnp=IsPronPropnPresent(sent,dep,MailTo,MailCC)  
                    #Get appreciated sentences with simple adjective noun pair
                    adjn= OnlyAdjNounPairPresent(dep,"Escalation")

                    #Get sentiment using Stanford core nlp
                    sentiment=getPositivityUsingSFCoreNLP(sent,dep)  
                    #Increase the score of Escalation by 0.5 if sentence is Positive or VeryPositive or Neutral.
                    if((sentiment in ['Negative','Verynegative']) and (len(avn)>0 or len(pwp)>0)):
                        escalationScore+=0.5
                    if(sentiment in ['Positive','Verypositive']):
                        escalationScore-=0.5
                    isNegWordExist,presentUsers=isValiedText(sent,dep,MailTo,MailCC,pwps)
                    
                    #Increase score by 0.5 if adjective modifier present
                    if(escalationScore>0):
                        isItEscalationText=True
                        if(isAdjModExist(dep,pwps)):
                            escalationScore+=0.5
                    adj=column(dep,4).count('ADJ') 
                    thanksWord=column(dep,6).count('thank') 
                    pobjects=[]
                    isThanksExist=isThanksPresent(sent)
                    #Reduce score for only thanks you text
                    if(isThanksExist and len(pwp)==1):
                        escalationScore=0
                        isAppreciated=False                        
                    seperateSentenceByInject=[]                    
                    NegSentiment=sentiment in ['Negative','Verynegative']
                    isNegativeWordPresent=False
                    fixedEscalation=False
                    
                    #Find the dependency in the adjective and other POS
                    pap=isPrepAdjPresent(dep,pwps)
                    vap=isVerbAppreciated(dep,pwps)
                    vrap=verRoot_adjPobjOfPrep(dep,pwps)
                    cal=conjAdjList(dep,pwps)
                    aap=amod_adj(dep,pwps,avn)
                    aav=attrAdj_verb(dep,pwps)
                    attr_adj_n=attr_adj_noun(dep,pwps,avn)
                    acr=adj_acomp_root(dep,pwps,avn)
                    ada=avn_dobj_adj_verb(dep,pwps,avn)
                    pva=pobj_verb_adj(dep,pwps,avn)
                    apam=avn_pobj_adj_amod(dep,pwps,avn)
                    if( pap or vap or vrap or cal or aap or aav or attr_adj_n or acr or ada or pva or apam):
                        fixedEscalation=True
                    metadeta=[]
                    
                    isAdjComp=isAdjCompoundWord(dep,pwps)
                    if(sentiment =='Negative'):
                        NegativeSents.append(sent)
                    if(sentiment in ['Negative','Neutral']):
                        isNegativeWordPresent=bool([d for d in dep if(d[1] in ['neg'])])
                    
                    lstmScore=LSTMCheck(sent,"Escalation")
                    
                    lstmScore=float(lstmScore[0][0])
                    if(escalationScore>=2 and lstmScore>0.6):
                        escalationScore-=2
                    if((fixedEscalation==False or escalationScore==0) and lstmScore<0.2):
                        if(escalationScore<2):
                            escalationScore+=2.1
                        fixedEscalation=True    
                    #Threshold score to between 0 and 5    
                    if(escalationScore>5):
                        escalationScore=5                        
                    if(escalationScore<0):                        
                        escalationScore=0
                    #print("adjn",adjn,"lstmScore","pwp",pwp,lstmScore,"escalationScore",escalationScore,"pap",pap," vap",vap,"vrap",vrap,"cal", cal," aap", aap,"aav",aav,"attr_adj_n", attr_adj_n,"acr", acr,"ada",ada,"pva", pva,"apam", apam)
                    if((NegSentiment or isNegativeWordPresent) and not isAdjComp):                        
                        escalationDF.append([MailBody,Appreciated_User,Mail_From,MailTo,MailCC,sent,avn,pwp,akp,pnp,adjn,sentiment,escalationScore,isItEscalationText,pobjects,lstmScore])
                    elif(adjn):
                        escalationDF.append([MailBody,Appreciated_User,Mail_From,MailTo,MailCC,sent,avn,pwp,akp,pnp,adjn,sentiment,escalationScore,isItEscalationText,pobjects,lstmScore])
                    
                    elif((NegSentiment or isNegativeWordPresent) and fixedEscalation and isNegWordExist):
                        escalationDF.append([MailBody,Appreciated_User,Mail_From,MailTo,MailCC,sent,avn,pwp,akp,pnp,adjn,sentiment,escalationScore,isItEscalationText,pobjects,lstmScore])                                                                
                    
                    if(not isThanksExist and escalationDF==[]):                        
                        escalationDF.append([MailBody,Appreciated_User,Mail_From,MailTo,MailCC,sent,avn,pwp,akp,pnp,adjn,sentiment,escalationScore,isItEscalationText,pobjects,lstmScore])                        
                    if(escalationDF==[]):                    
                        escalationScore=0
                        escalationDF.append([MailBody,Appreciated_User,Mail_From,MailTo,MailCC,sent,avn,pwp,akp,pnp,adjn,sentiment,escalationScore,False,pobjects,lstmScore])                    
                    if(escalationScore>0 and not fixedEscalation and not adjn):                    
                        isAppreciated=True
                        compoundNounsDep = [x for x in dep if ((x[0] in avn or x[2] in avn) and x[1] == 'compound')]                       
                        if(len(avn)==1 and len(compoundNounsDep)==1):
                            isAppreciated=False
                        if(len(avn)==0):
                            isAppreciated=False                    
                        if(isAppreciated and len(escalationDF)>0 and escalationDF[len(escalationDF)-1][5] == sent):
                            escalationScore=0
                            escalationDF[len(escalationDF)-1][13]=False
                            escalationDF[len(escalationDF)-1][12]=0
                    if(escalationScore>0):
                        allPwps_Avns.append([escalationDF[len(escalationDF)-1][5],pwps,avn,escalationScore])                        
        return escalationDF       
    
#check PRON or PROPN present or not 
def IsPronPropnPresent(sent,dep,mail_To,mail_CC):
    isProNounPresent=any(x.lower() in propNouns for x in sent.split(' '))    
    if(len(check_PROPNPresent(dep,mail_To,mail_CC)[0])==0 and not isProNounPresent):
        return False
    else:
        return True

#Clean the text to remove html or javascript or any other tags.
class MLStripper(HTMLParser):
    def __init__(self):
        # initialize the base class
        HTMLParser.__init__(self)

    def read(self, data):
        # clear the current output before re-use
        self._lines = []
        # re-set the parser's state before re-use
        self.reset()
        self.feed(data)
        return ''.join(self._lines)

    def handle_data(self, d):
        self._lines.append(d)


#Find is the text conain any URL        
def FindURL(string): 
    # findall() has been used  
    # with valid conditions for urls in string 
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string) 
    return url 
      
def strip_tags(html):
    s = MLStripper()
    return s.read(html)

#Filter the text with unwanted characters or tags and split with \n
def clean_text(text):    
    text=text.replace('"', '')
    text=text.replace("<br>","\n")
    text=re.sub('[^A-Za-z.,-]+\<>*', ' ', strip_tags(text)).strip()
    text=text.split("-----Original Message-----")[0]         
    for s in replace_list:        
        text = text.replace(s, replace_list[s])    
    
    if(text.find(',')>-1):
        i=text.index(",")
        if(len(text[i])>i and text[i+1]=='\n'):
            text.replace(",","")        
    
    text=text.replace("<div>","\\n").replace("</div>","").replace("&"," and ").replace("\r","").replace('\\u2019',' a').replace("\n\n","\n").replace("\n \n","\n").replace("\t","").replace("Thanks","thank you").replace("Thankx","thank you").replace("thanks","thank you").replace("-","").replace("\\n","\n").replace('!','').replace(':)','great').replace('(:','bad').split("\n")
    text=[m.strip() for m in text]
    i=-1    
    while(i!=len(text)):
        i+=1
        if(i<len(text)-1 and len(text[i+1])>0 and text[i]!='' and text[i][-1] !='.'):
            text[i]+=' '+text[i+1]        
            del text[i+1]
            i=i-1
    text=[m.strip() for m in text if(len(m)>3)]
    
    for k in range(len(text)):
        if(k<len(text)-1 and text[k+1][0].islower()):
            text[k]+=' '+text[k+1]
            del text[k+1]
    
    return text

#Check that text is not empty.
def checkApiParameters(mailBody,MailFrom,MailTo,MailCC):
    if type(MailFrom)!=str:
        MailFrom=""
    if type(MailTo)!=str:
        MailTo=""
    if type(MailCC) !=str:
        MailCC=""
    if type(mailBody)!=str:
        mailBody=""
    return mailBody,MailFrom,MailTo,MailCC

#Filter the signature from the text
def getSignatureSeperatedMail(mailBody,MailFrom):    
    sigStart=[ele for ele in mailBody if(len(ele.strip()) >0 and (ele.strip().lower()[-1] is '.' 
                                                          and ele.strip().lower()[:-1]  in sampleSignatures) 
                                         or (ele.strip().lower()[:-1] is not '.' and ele.strip().lower()[:]  
                                             in sampleSignatures))] 
    
    if(len(sigStart)>0):
        mailBody=mailBody[:mailBody.index(sigStart[len(sigStart)-1])]
    fromUser=[e for e in mailBody if(getMailSignature(e,MailFrom) == True)]
    fromUser=[f for f in fromUser if f!='']   
    if(len(fromUser)>0):
        mailBody=mailBody[:mailBody.index(fromUser[len(fromUser)-1])]
    return mailBody

#Calculate the average score against the multiple appreciated records.
def getAverageScore(escalationsDF):
    avgScore=0
    if(escalationsDF.shape[0]>0):
        avgScore=float(escalationsDF.loc[:,"Escalation Score"].values.max())
        if(avgScore>3 and escalationsDF.loc[:,"Escalation Score"].values.sum()>=7):
            avgScore=5
        elif(avgScore>4 and escalationsDF.loc[:,"Escalation Score"].values.sum()<=7):
            avgScore=4.5
        elif(avgScore>4 and escalationsDF.loc[:,"Escalation Score"].values.sum()<=6):
            avgScore=4
    return avgScore

#Calculate the average score against the multiple appreciated records.
def getAverageAppreciationScore(appreciationDF):
    avgScore=0
    if(appreciationDF.shape[0]>0):
        avgScore=float(appreciationDF.loc[:,"Appreciation Score"].values.max())
        if(avgScore>3 and appreciationDF.loc[:,"Appreciation Score"].values.sum()>=7):
            avgScore=5
        elif(avgScore<4 and appreciationDF.loc[:,"Appreciation Score"].values.sum()>=7):
            avgScore=4.5
        elif(avgScore<4 and appreciationDF.loc[:,"Appreciation Score"].values.sum()>=6):
            avgScore=4
    return avgScore

#Convert dataframe object to complex json object
def DataframeToJSON(escalationsDF,avgScore,userScoreMapping):
    jsonArray=[]
    EscalationDataJson={}

    if(escalationsDF.shape[0]>0):            
        EscalationDataJson={
        "TextFrom":escalationsDF.loc[0:0,"MailFrom"].values[0],
        "TextTo":escalationsDF.loc[0:0,"MailTO"].values[0],
        "TextCC":escalationsDF.loc[0:0,"MailCC"].values[0],
        "Text":escalationsDF.loc[0:0,"MailBody"].values[0],
        "AverageScore":avgScore
        }    
    
    for i in range(escalationsDF.shape[0]):
        sentData={}
        sentData["EscalatedUser"]=escalationsDF.loc[i:i,"EscalatedUser"].values[0]
        sentData["EscalationScore"]=float(escalationsDF.loc[i:i,"Escalation Score"].values[0])
        sentData["EscalatedSentence"]=escalationsDF.loc[i:i,"Sentence"].values[0]
        sentData["isItEscalationText"]=bool(escalationsDF.loc[i:i,"isItEscalationText"].values[0])
        #sentData["IsAppreciated_ObjectPresent"]=escalationsDF.loc[i:i,"Object"].values[0]
        sentData["lstmScore"]=escalationsDF.loc[i:i,"lstmScore"].values[0]
        jsonArray.append(sentData)
    RecipientUserScoreMapping=[]
    for user in userScoreMapping:
        userwiseScore={}
        userwiseScore["RecipientUser"]=user
        averageUserScore=float(userScoreMapping[user])
        if(float(userScoreMapping[user])>=7):
            averageUserScore=5.0
        if(float(userScoreMapping[user])>=5 and float(userScoreMapping[user])<7):
            averageUserScore=4.0
            
        userwiseScore["EscalationScore"]=float(averageUserScore)
        RecipientUserScoreMapping.append(userwiseScore)
    EscalationDataJson["EscalationDetails"]=jsonArray
    EscalationDataJson["RecipientUserEscalationDetails"]=RecipientUserScoreMapping
    
    return EscalationDataJson

#Read labled emails from the csv file for training the model. 
def readData():
    df=pd.read_csv("IMDB_BravoDataSet.csv")    
    df=df[df.Text!=3]
    df=df.dropna()
    df=df.drop_duplicates()
    df.reset_index(inplace = True)
    df.drop(['index'],axis=1,inplace = True)
    return df

#Split data for train test model
def splitData():
    df=readData()
    X = df.Text
    Y = df.Label
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1,1)
    
    #Split data to train test objects
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)    
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
    return tok,sequences,sequences_matrix,Y_train

#Initialise the layers for the LSTM model.
def LSTMTraining():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

#Train the model using LSTM
def trainModel():
    tok,sequences,sequences_matrix,Y_train=splitData()

    #save the tok file to disk for furhter use
    pickle.dump(tok, open(filenameTok, 'wb'))
    model = LSTMTraining()
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    #Train model with training data
    model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
    return model

#Save model for further use
def saveModel(): 
    model=trainModel()  
    #Save model to disk
    pickle.dump(model, open(filename, 'wb'))

#Classify the text using LSTM in appreciated or non appreciated text
def LSTMCheck(Text,mod):
    max_words = 15000
    max_len = 300
    
    X_test=pd.Series([Text])
    if(mod=="Escalation"):
        test_sequences = tok.texts_to_sequences(X_test)
        test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    
        #Predict the accuracy score on new text
        with graph.as_default():
            return str(model.predict(test_sequences_matrix)[0][0])
    else:
        test_sequences = tok_Appr.texts_to_sequences(X_test)
        test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    
        #Predict the accuracy score on new text
        with graph.as_default():
            return str(model_Appr.predict(test_sequences_matrix)[0][0])
        

#Logic to get is the text has dependency matrix which proves that text is appreciated text or not. also cross check against LSTM to get extream appreciated texts and neutral text         
def checkAppreciationText(Appreciated_User,MailTo,MailCC,Mail_From,Sentence,MailBody):
    dep=getDependency(Sentence,"Appreciation")  
    isAdjExist=column(dep,4).count('ADJ')   
    
    #Check is the appreciation keywords present in the text
    isApprKeyPresent=isAppreciateKeywordPresent(dep)
    
    #Reject the sentences without any adjectives
    if(isAdjExist>0 or isApprKeyPresent):        
        #Get the seperate simple sentences from the complex text.
        sents=seperateSent(Sentence,dep)        
        ApprDF=[]
        for j in range(len(sents)):
            aprScore=0
            isItAppriciationText=False
            sent=sents[j].strip()           
            if(sent!='' and filterDataType(sent)):  
                sent=re.sub(r"[^a-zA-Z.,-]+", ' ', sent).strip()
                dep=getDependency(sent,"Appreciation")

                #Get the imperative sentences           
                isImperativeSent=getActionRequiredText(sent,dep)

                onlyPropNounsInSent=onlyPropNounsInSents(dep)   
                #Do not Process further if isImperative is true or sentence only contains the proper nouns             
                if((not isImperativeSent and not onlyPropNounsInSent) or isApprKeyPresent):
                    adj=column(dep,4).count('ADJ') 
                    if(adj>1):
                        aprScore=adjustMultipleAdjectives(dep,aprScore)                                            
                    avn=isApprVerbNounPresent(dep)
                    pwp=isPossitiveWordPresent(dep,"Appreciation")
                    pwps=[]
                    pwps=[p[0].lower() for p in pwp]
                   
                    if(len(pwps)>0 and set(avn)==set(pwps)):
                        avn=[]
                        avn_pwpNotSame=False
                        
                    if(len(pwp)>0):
                        aprScore+=pd.DataFrame(pwp).sort_values(1,ascending=False)[1].max()
                    akp=isAppreciateKeywordPresent(dep)
                    pnp=IsPronPropnPresent(sent,dep,MailTo,MailCC)

                    #Get appreciated sentences with simple adjective noun pair
                    adjn= OnlyAdjNounPairPresent(dep,"Appreciation")

                    #Get sentiment using Stanford core nlp
                    sentiment=getPositivityUsingSFCoreNLP(sent,dep)  
                                        
                    #Increase the score of appreciation by 0.5 if sentence is Positive or VeryPositive or Neutral.
                    if((sentiment in ['Positive','Verypositive','Neutral']) and (len(avn)>0 or len(pwp)>0 or akp)):
                        aprScore+=0.5
                    
                    isPossWordExist,presentUsers=isValiedText(sent,dep,MailTo,MailCC,pwps)
                    
                    #Increase score by 0.5 if adjective modifier present
                    if(aprScore>0):
                        isItAppriciationText=True
                        if(isAdjModExist(dep,pwps)):
                            aprScore+=0.5
                    
                    adj=column(dep,4).count('ADJ') 
                    thanksWord=column(dep,6).count('thank') 
                    pobjects=[]
                    isThanksExist=isThanksPresent(sent)
                    #Reduce score for only thanks you text
                    if(isThanksExist and len(pwp)==1):
                        aprScore=0
                        isAppreciated=False                        
                        
                    seperateSentenceByInject=[]                    
                    posSentiment=sentiment in ['Positive','Verypositive','Neutral']
                    isNegativeWordPresent=False
                    fixedAppr=False
                    
                    #Find the dependency in the adjective and other POS
                    pap=isPrepAdjPresent(dep,pwps)
                    vap=isVerbAppreciated(dep,pwps)
                    vrap=verRoot_adjPobjOfPrep(dep,pwps)
                    cal=conjAdjList(dep,pwps)
                    aap=amod_adj(dep,pwps,avn)
                    aav=attrAdj_verb(dep,pwps)
                    attr_adj_n=attr_adj_noun(dep,pwps,avn)
                    acr=adj_acomp_root(dep,pwps,avn)
                    ada=avn_dobj_adj_verb(dep,pwps,avn)
                    pva=pobj_verb_adj(dep,pwps,avn)
                    apam=avn_pobj_adj_amod(dep,pwps,avn)
              
                    
                    if( pap or vap or vrap or cal or aap or aav or attr_adj_n or acr or ada or pva or apam):
                        fixedAppr=True
                    metadeta=[]
                    
                    isAdjComp=isAdjCompoundWord(dep,pwps)
                    if(sentiment =='Negative'):
                        NegativeSents.append(sent)
                    if(sentiment in ['Negative','Neutral']):
                        isNegativeWordPresent=bool([d for d in dep if(d[1] in ['neg'])])
                    #print("sent",sent)
                    lstmScore=LSTMCheck(sent,"Appreciation")
                    
                    lstmScore=float(lstmScore)
                    #print("lstmScore",lstmScore)
                    #print("lstmScore","adjn",adjn,"pwp",pwp,lstmScore,"aprScore",aprScore,"pap",pap," vap",vap,"vrap",vrap,"cal", cal," aap", aap,"aav",aav,"attr_adj_n", attr_adj_n,"acr", acr,"ada",ada,"pva", pva,"apam", apam)
                    
                    if(aprScore>=2 and lstmScore<0.2 and not akp):
                        aprScore-=2
                    if((fixedAppr==False or aprScore==0) and lstmScore>0.9):
                        if(aprScore<2):
                            aprScore+=2.1
                        fixedAppr=True    
                    if(akp and aprScore==2):
                        aprScore+=0.5
                    #Threshold score to between 0 and 5    
                    if(aprScore>5):
                        aprScore=5                        
                    if(aprScore<0):                        
                        aprScore=0
                   
                    if((posSentiment or not isNegativeWordPresent) and akp and not isAdjComp):                        
                        ApprDF.append([MailBody,Appreciated_User,Mail_From,MailTo,MailCC,sent,avn,pwp,akp,pnp,adjn,sentiment,aprScore,isItAppriciationText,pobjects,lstmScore])
                    elif(adjn):
                        ApprDF.append([MailBody,Appreciated_User,Mail_From,MailTo,MailCC,sent,avn,pwp,akp,pnp,adjn,sentiment,aprScore,isItAppriciationText,pobjects,lstmScore])
                    
                    elif((posSentiment or not isNegativeWordPresent) and fixedAppr and isPossWordExist):
                        ApprDF.append([MailBody,Appreciated_User,Mail_From,MailTo,MailCC,sent,avn,pwp,akp,pnp,adjn,sentiment,aprScore,isItAppriciationText,pobjects,lstmScore])                                                                
                    
                    if(isThanksExist and ApprDF==[]):                        
                        ApprDF.append([MailBody,Appreciated_User,Mail_From,MailTo,MailCC,sent,avn,pwp,akp,pnp,adjn,sentiment,aprScore,isItAppriciationText,pobjects,lstmScore])                        
                    if(ApprDF==[]):                    
                        aprScore=0
                        ApprDF.append([MailBody,Appreciated_User,Mail_From,MailTo,MailCC,sent,avn,pwp,akp,pnp,adjn,sentiment,aprScore,False,pobjects,lstmScore])                    
                    if(aprScore>0 and not fixedAppr and not akp and not adjn):                    
                        isAppreciated=True
                        compoundNounsDep = [x for x in dep if ((x[0] in avn or x[2] in avn) and x[1] == 'compound')]                       
                        if(len(avn)==1 and len(compoundNounsDep)==1):
                            isAppreciated=False
                        if(len(avn)==0):
                            isAppreciated=False                    
                        if(not isAppreciated and not isThanksExist and len(ApprDF)>0 and ApprDF[len(ApprDF)-1][5] == sent):
                            aprScore=0
                            ApprDF[len(ApprDF)-1][13]=False
                            ApprDF[len(ApprDF)-1][12]=0
                    if(aprScore>0):
                        allPwps_Avns.append([ApprDF[len(ApprDF)-1][5],pwps,avn,aprScore])                        
        return ApprDF  

    
def getAppreciationOrNotMatrix(ds,MailTo,MailCC,MailFrom,sentence,originalMail):        
    try:
        data=[]
        #Iterate all the sentences in the mail
        for j in range(len(ds)):            
            for appreciated_Emp, sentence in json.loads(ds[j]).items(): 
                sentence=sentence.strip()                
                appreciation=checkAppreciationText(appreciated_Emp,MailTo,MailCC,MailFrom,sentence,originalMail)
                #detailedSents.extend(seperatedSents)       
                if(appreciation != []):
                    data.append(appreciation)
    
        appreciationDF=pd.DataFrame()
        for i in range(len(data)):       
            appreciationDF=appreciationDF.append(pd.DataFrame(data[i],columns=['MailBody','AppreciatedUser','MailFrom','MailTO','MailCC', 'Sentence','avn','pwp','akp','pnp','adjn','Sentiment','Appreciation Score','IsItAppreciation','Object','lstmScore']))
        
        appreciationDF=appreciationDF.reset_index()        
        avgScore=0.0
        if(appreciationDF.shape[0]>0):
            avgScore=getAverageAppreciationScore(appreciationDF)
            
        return avgScore
    
    except Exception as e:
        logger.debug(traceback.print_exc())
        tc.track_exception()
        print(traceback.print_exc())
        tc.flush()
        logging.shutdown()
        return 0  


@app.route('/api/intent/Escalation', methods=['POST','GET'])
def getEscalationOrNotMatrix():        
    try:
        
        if request.method == 'POST':
            req_data = request.get_json(force=True)        
            mailBody=req_data['Text']
            MailFrom=req_data['TextFrom']
            MailTo=req_data['TextTo']
            MailCC=req_data['CC']    
        else:    
            parms=request.args.to_dict() 
            mailBody=parms['Text'] 
            MailFrom=parms['TextFrom'] 
            MailTo=parms['TextTo'] 
            MailCC=parms['CC']          
        
        log='Escalation Detection API Parameters: Text:'+str(mailBody)+" TO:"+str(MailTo)+" CC:"+str(MailCC)
        logger.debug(log)
        originalMail=mailBody    

        #Clean the parameters which are empty        
        mailBody,MailFrom,MailTo,MailCC=checkApiParameters(mailBody,MailFrom,MailTo,MailCC)               
        mailBody=clean_text(mailBody) 
       
        detailedSents=[]
        escalationDF=data=Escalation=mailSignatures=[]        
        
        #Filter out the signature from the text
        mailBody=getSignatureSeperatedMail(mailBody,MailFrom)
        
        for i in range(len(mailBody)):   
            url=FindURL(mailBody[i])            
            if(len(url)>0):
                mailBody[i]=mailBody[i].replace(url[0],'')
            seperatedSents=mailBody[i].split('.')
       
            for i in range(len(seperatedSents)):            
                if(len(seperatedSents)>i+1 and len(seperatedSents[i+1])<3):
                    seperatedSents[i]+=seperatedSents[i+1]
            detailedSents.extend(seperatedSents)
            
        #Extract the detailsed structer of the text including targeted user.
        ds=findSentanceUserMapping(detailedSents,MailFrom,MailTo,MailCC)    
        EscalationDataJson={}
        
        
            #Iterate all the sentences in the mail
        for j in range(len(ds)):          
            for appreciated_Emp, sentence in json.loads(ds[j]).items(): 
                sentence=sentence.strip()    
                appreciationScore=getAppreciationOrNotMatrix(ds,MailTo,MailCC,MailFrom,sentence,originalMail)
                #print("appreciationScore",appreciationScore)
                if(appreciationScore<2):
                    Escalation=checkEscalationText(appreciated_Emp,MailTo,MailCC,MailFrom,sentence,originalMail)
                    detailedSents.extend(seperatedSents)       
                    if(Escalation != []):
                        data.append(Escalation)
        
        escalationsDF=pd.DataFrame()
        for i in range(len(data)):
            escalationsDF=escalationsDF.append(pd.DataFrame(data[i],columns=['MailBody','EscalatedUser','MailFrom','MailTO','MailCC', 'Sentence','avn','pwp','akp','pnp','adjn','Sentiment','Escalation Score','isItEscalationText','Object','lstmScore']))
        
        escalationsDF=escalationsDF.reset_index()
            
        groupedScore={}
        avgScore=0.0
        userScoreMapping={}
        
        if(escalationsDF.shape[0]>0):
            avgScore=getAverageScore(escalationsDF)

            #Group by userwise score and get sum of score
            groupedScore=escalationsDF.groupby(['EscalatedUser'])['Escalation Score'].sum()
            
            for names,score in groupedScore.iteritems():
                name=names.split(";")
                for n in name:
                    if n.strip() in userScoreMapping:
                        userScoreMapping[n.strip()]=userScoreMapping[n.strip()]+score
                    else:
                        userScoreMapping[n.strip()]=score
        EscalationDataJson=DataframeToJSON(escalationsDF,avgScore,userScoreMapping)
        logger.debug("Response Returned Success")
        tc.flush()
        return json.dumps(EscalationDataJson)
    
    except Exception as e:
        logger.debug(traceback.print_exc())
        tc.track_exception()
        tc.flush()
        logging.shutdown()
        return json.dumps({"Error":"Error has occured, Contact to Administrator."})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',threaded=False)