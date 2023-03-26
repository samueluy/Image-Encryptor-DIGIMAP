from django.shortcuts import render
from . forms import image


# Create your views here.
def home(request): # render home.html
    if request.method == 'POST':
        form = image(request.POST, request.FILES)
        if form.is_valid():
            # get the uploaded image from form
            image_file = form.cleaned_data['img']
            print(image_file)

            # render image
            return render(request, 'display_image.html', {'img': image_file})
        else:
            form = image()
            print(":yhere")
            return render(request, 'upload_image.html', {'form': form})
        
    else:
        form = image()
        return render(request, 'upload_image.html', {'form': form})