from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'home.html')


def button1(request):
    return render(request, 'button1.html')


def button2(request):
    return render(request, 'button2.html')

