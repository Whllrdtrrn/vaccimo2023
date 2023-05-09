import django_filters
from django_filters import DateFilter, ModelChoiceFilter, BooleanFilter, CharFilter, NumberFilter, ChoiceFilter
from .models import *
from django.forms import DateInput
from django import forms


## information collection
class OrderFilter(django_filters.FilterSet):
    VACCINE_BRAND_CHOICES = (
        ('', 'All'),
        ('pfizer', 'Pfizer'),
        ('moderna', 'Moderna'),
        ('johnson_and_Johnsons', 'johnson and Johnsons'),
        ('sinovac', 'Sinovac'),
        ('astraZeneca', 'AstraZeneca'),
    )
    start_date = DateFilter(field_name="date_created", lookup_expr='gte')
    name = CharFilter(field_name="name", lookup_expr='icontains')
    age = NumberFilter(field_name="age", lookup_expr='exact')
    gender = CharFilter(field_name="gender", lookup_expr='exact')
    vaccine_brand = ChoiceFilter(field_name="vaccination_brand", choices=VACCINE_BRAND_CHOICES)
    class Meta:
        model = user
        fields = ['name', 'vaccine_brand', 'age', 'gender', 'start_date', 'vaccination_brand']
        exclude = ['author','date_created']
        
        
class EffectFilter(django_filters.FilterSet):
    SYMPTOMS = (
        ('', 'All'),
        ('mild', 'mild'),
        ('Moderate', 'Moderate'),
        ('Severe', 'Severe'),
    )
    start_date = DateFilter(field_name="warmth", lookup_expr='gte')
    feverish = ChoiceFilter(field_name="feverish", choices=SYMPTOMS)
    author = ModelChoiceFilter(queryset=user.objects.all())

    class Meta:
        model = sideeffect
        fields = ['start_date', 'muscle_ache','headache','fever','redness','swelling','tenderness','warmth','itch','induration','feverish','chills','join_pain','fatigue','nausea','vomiting','redness']
        exclude = ['author','warmth'] 
