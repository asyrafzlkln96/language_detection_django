from django.shortcuts import render
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound, JsonResponse
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt


# ML stuff 
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Import to ML (scikit-learn) Libraries
from sklearn.linear_model import RidgeClassifier #RidgeClassifier

# Create your views here.
class LanguageDetect(APIView):
    @csrf_exempt
    def post(self, request, format=None):
        data = request.data
        text = data["Text"]
        df = pd.read_csv('language_detection.csv')
        X_train, X_test, y_train, y_test = train_test_split(df.Text, 
                                                    df.language,
                                                    test_size=0.325000000000000001,
                                                    random_state=2551,
                                                    shuffle=True)
        text = [text]

        X_CountVectorizer = CountVectorizer(stop_words='english')
        X_train_counts = X_CountVectorizer.fit_transform(X_train)
        X_TfidfTransformer = TfidfTransformer()

        X_train_tfidf = X_TfidfTransformer.fit_transform(X_train_counts)
        text_counts = X_CountVectorizer.transform(text)

        with open('language_detection_Ridge_CLF.pkl', 'rb') as file:  
            lang_det_ridge_model = pickle.load(file)

        prediction = lang_det_ridge_model.predict(text_counts)

        data = {'Status': '200', 'Language Detected': prediction[0]}
        return JsonResponse(data, status=status.HTTP_200_OK)

class TrainML(APIView):
    @csrf_exempt
    def post(self, request, format=None):
        pass


