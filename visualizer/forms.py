from django import forms
from crispy_forms.helper import FormHelper
class TestData(forms.Form):
    customer_nodes = forms.IntegerField(label = 'Customer Nodes', min_value = 5, max_value = 50)
    depot_nodes = forms.IntegerField(label = 'Depot Nodes', min_value = 1, max_value = 10)
    max_vehicles = forms.IntegerField(label = 'Vehicles',min_value = 1, max_value = 10)
    iterations = forms.IntegerField(label = 'Iterations', min_value = 1, max_value = 50)
    options = (
        ("2","2"),
        ("4","4"),
        ("6","6"),
        ("8","8"),
        ("10","10"),
    )
    P = forms.ChoiceField(label = 'Parameter 1', choices = options)
    max_no_improve = forms.IntegerField(label = 'Parameter 2', min_value = 2, max_value = 5)
   
    Coordinate = forms.CharField(widget=forms.Textarea, label='Coordinates')
