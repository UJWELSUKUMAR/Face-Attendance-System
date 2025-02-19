from django.urls import path
from home.views import *

urlpatterns = [
    path('',index,name='index'),
    path('second',second,name='second'),
    path('video_stream/', video_stream, name='video_stream'),
    path("add_person",add_person,name="add_person")
]

