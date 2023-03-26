from django import forms

class image(forms.Form):
    img = forms.ImageField(initial=None)