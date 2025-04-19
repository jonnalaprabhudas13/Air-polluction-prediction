from django import forms

class AirQualityForm(forms.Form):
    temperature = forms.FloatField(label='Temperature (°C)', min_value=-50, max_value=60)
    humidity = forms.FloatField(label='Humidity (%)', min_value=0, max_value=100)
    pm25 = forms.FloatField(label='PM2.5 (µg/m³)', min_value=0)
    pm10 = forms.FloatField(label='PM10 (µg/m³)', min_value=0)
    no2 = forms.FloatField(label='NO2 (ppb)', min_value=0)
    so2 = forms.FloatField(label='SO2 (ppb)', min_value=0)
    co = forms.FloatField(label='CO (ppm)', min_value=0)
    proximity = forms.FloatField(label='Proximity to Industrial Areas (km)', min_value=0)
    population_density = forms.IntegerField(label='Population Density (people/km²)', min_value=0)