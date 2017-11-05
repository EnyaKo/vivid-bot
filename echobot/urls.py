from django.conf.urls import url

from . import views

urlpatterns = [
    url('^callback/', views.callback),
    url('^direct_callback/', views.direct_callback),
]
