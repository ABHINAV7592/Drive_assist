# detection/routing.py
from django.urls import re_path
from .consumers import TrafficDetectionConsumer, PotholeDetectionConsumer, DrowsinessCameraConsumer




websocket_urlpatterns = [
    re_path(r'ws/traffic-detection/$', TrafficDetectionConsumer.as_asgi()),
    re_path(r'ws/pothole-detection/$', PotholeDetectionConsumer.as_asgi()),
    re_path(r"ws/DrowsinessCamera/$", DrowsinessCameraConsumer.as_asgi()),
]

