from django.urls import path
from django.conf.urls import include, url

from . import views

app_name = 'language_detect'

urlpatterns = [
    # url(r"^train", views.LanguageDetect.train_language.as_view(), name="train_language"),
    url(r"^predict", views.LanguageDetect.as_view()),
]