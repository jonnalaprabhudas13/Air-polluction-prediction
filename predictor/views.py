from django.shortcuts import render
from django.http import HttpResponseServerError
from .forms import AirQualityForm
from .models import AirQualityPredictor

def predict_air_quality(request):
    try:
        predictor = AirQualityPredictor()
        result = None
        
        if request.method == 'POST':
            form = AirQualityForm(request.POST)
            if form.is_valid():
                input_data = {
                    'temperature': form.cleaned_data['temperature'],
                    'humidity': form.cleaned_data['humidity'],
                    'pm25': form.cleaned_data['pm25'],
                    'pm10': form.cleaned_data['pm10'],
                    'no2': form.cleaned_data['no2'],
                    'so2': form.cleaned_data['so2'],
                    'co': form.cleaned_data['co'],
                    'proximity': form.cleaned_data['proximity'],
                    'population_density': form.cleaned_data['population_density']
                }
                result = predictor.predict(input_data)
        else:
            form = AirQualityForm()
        
        return render(request, 'predictor/predict.html', {
            'form': form,
            'result': result
        })
    
    except FileNotFoundError as e:
        return HttpResponseServerError(
            "Dataset file not found. Please ensure the CSV file is in the predictor/data directory."
        )
    except Exception as e:
        return HttpResponseServerError(
            f"An error occurred: {str(e)}"
        )
















# from django.shortcuts import render

# # Create your views here.

# from django.shortcuts import render
# from .forms import AirQualityForm
# from .models import AirQualityPredictor

# def predict_air_quality(request):
#     predictor = AirQualityPredictor()
#     result = None
    
#     if request.method == 'POST':
#         form = AirQualityForm(request.POST)
#         if form.is_valid():
#             input_data = {
#                 'temperature': form.cleaned_data['temperature'],
#                 'humidity': form.cleaned_data['humidity'],
#                 'pm25': form.cleaned_data['pm25'],
#                 'pm10': form.cleaned_data['pm10'],
#                 'no2': form.cleaned_data['no2'],
#                 'so2': form.cleaned_data['so2'],
#                 'co': form.cleaned_data['co'],
#                 'proximity': form.cleaned_data['proximity'],
#                 'population_density': form.cleaned_data['population_density']
#             }
#             result = predictor.predict(input_data)
#     else:
#         form = AirQualityForm()
    
#     return render(request, 'predictor/predict.html', {
#         'form': form,
#         'result': result
#     })