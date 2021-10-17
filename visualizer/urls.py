from django.urls import path

from . import views

urlpatterns = [
    path('', views.index,name = "index"),
    path('graph',views.graph_view,name = "graph"),
    path('ajax_posting', views.ajax_posting, name='ajax_posting'),
]
