from flask import render_template,flash, request
from chatbot.forms import chatbotform
from chatbot.__init__ import model,words,classes,intents
from flask import Flask
from flask_ngrok import run_with_ngrok
import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential,load_model
import random
from datetime import datetime
import pytz
from alpha_vantage.timeseries import TimeSeries 
import requests
import os
import billboard
import time
from pygame import mixer
import COVID19Py
from flask import Flask, render_template, jsonify, request

from yahoo_fin import stock_info as si


from transformers import pipeline

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

app = Flask(__name__)
run_with_ngrok(app)


#Predict
def clean_up(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[ lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def create_bow(sentence,words):
    sentence_words=clean_up(sentence)
    bag=list(np.zeros(len(words)))

    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence,model):
    p=create_bow(sentence,words)
    res=model.predict(np.array([p]))[0]
    threshold=0.8
    results=[[i,r] for i,r in enumerate(res) if r>threshold]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]

    for result in results:
        return_list.append({'intent':classes[result[0]],'prob':str(result[1])})
    return return_list

def get_response(return_list,intents_json,text):

    if len(return_list)==0:
        tag='noanswer'
    else:
        tag=return_list[0]['intent']
    if tag=='datetime':
        x=''
        tz = pytz.timezone('Asia/Kolkata')
        dt=datetime.now(tz)
        x+=str(dt.strftime("%A"))+' '
        x+=str(dt.strftime("%d %B %Y"))+' '
        x+=str(dt.strftime("%H:%M:%S"))
        return x,'datetime'



    if tag=='weather':
        x=''
        api_key='987f44e8c16780be8c85e25a409ed07b'
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        # city_name = input("Enter city name : ")
        city_name = text.split(':')[1].strip()
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url)
        response=response.json()
        pres_temp=round(response['main']['temp']-273,2)
        feels_temp=round(response['main']['feels_like']-273,2)
        cond=response['weather'][0]['main']
        x+='Present temp.:'+str(pres_temp)+'C. Feels like:'+str(feels_temp)+'C. '+str(cond)
        print(x)
        return x,'weather'

    if tag=='news':
        main_url = " http://newsapi.org/v2/top-headlines?country=in&apiKey=bc88c2e1ddd440d1be2cb0788d027ae2"
        open_news_page = requests.get(main_url).json()
        article = open_news_page["articles"]
        results = []
        x=''
        for ar in article:
            results.append([ar["title"],ar["url"]])

        for i in range(10):
            x+=(str(i + 1))
            x+='. '+str(results[i][0])
            x+=(str(results[i][1]))
            if i!=9:
                x+='\n'

        return x,'news'

    if tag=='cricket':
        c = Cricbuzz()
        matches = c.matches()
        for match in matches:
            print(match['srs'],' ',match['mnum'],' ',match['status'])





    if tag=='covid19':

        covid19=COVID19Py.COVID19(data_source='jhu')
        country=text.split(':')[1].strip()
        x=''
        if country.lower()=='world':
            latest_world=covid19.getLatest()
            x+='Confirmed Cases:'+str(latest_world['confirmed'])+' Deaths:'+str(latest_world['deaths'])
            return x,'covid19'
        else:
            latest=covid19.getLocations()
            latest_conf=[]
            latest_deaths=[]
            for i in range(len(latest)):

                if latest[i]['country'].lower()== country.lower():
                    latest_conf.append(latest[i]['latest']['confirmed'])
                    latest_deaths.append(latest[i]['latest']['deaths'])
            latest_conf=np.array(latest_conf)
            latest_deaths=np.array(latest_deaths)
            x+='Confirmed Cases:'+str(np.sum(latest_conf))+' Deaths:'+str(np.sum(latest_deaths))
            return x,'covid19'


    if tag=='stock_info':
    	stock= text.split(':')[1].strip()
    	x=''
    	result=si.get_live_price(stock)
    	x+='The current price is '+ "{0:.2f}".format(result)
    	return x,'stock_info'

    else:
    	if len(text)<=5:
    		list_of_intents= intents_json['intents']
    		for i in list_of_intents:
    			if tag==i['tag'] :
    				result= random.choice(i['responses'])
    		return result,tag
    	
    	else:
    		question_answering = pipeline("question-answering")
    		context = """Machine learning (ML) is the study of computer algorithms that improve automatically through experience. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks."""
    		question = request.args.get('msg')
    		result = question_answering(question=question, context=context)
    		return result['answer'],'para'




    


  

def response(text):
    return_list=predict_class(text,model)
    response,_=get_response(return_list,intents,text)
    return response




@app.route('/chat',methods=["POST","GET"])



#@app.route('/')
def func():
    msg = request.args['msg']
    file1 = open("MyFile.txt","a")
    file1.writelines("Me:  "+msg+ "\n")
    resp=response(msg)
    file1.writelines("Bot:  "+resp+"\n")
    return resp


if __name__ == "__main__":
    app.run()