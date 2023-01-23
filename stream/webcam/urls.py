from django.contrib import admin
from django.urls import path,include
from webcam import views
urlpatterns = [
    path('login/', views.login, name='login'),
    path('', views.index, name='index'),
    path('video_feed', views.video_feed, name='video_feed')

]
