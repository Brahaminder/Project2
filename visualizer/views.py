from django.shortcuts import render
from django.http import HttpResponse
from django.core.exceptions import ValidationError
from django.http import JsonResponse

from .forms import TestData

# Create your views here.
from .MVRP import Model,TSP



    

def index(request):
    if(request.method == 'GET'):
        form = TestData()
        return render(request,'Home.html',{'form':form})


def graph_view(request):
    return render(request,'mygraph.html')


def ajax_posting(request):
    if request.is_ajax():
        try:
            n = int(request.POST['customer_nodes'])
            m = int(request.POST['depot_nodes'])
            max_vehicles = int(request.POST['max_vehicles'])
            iterations = int(request.POST['iterations'])
            bucket_size = int(request.POST['bucket_size'])
            max_no_improve = int(request.POST['max_no_improve'])
            coordinates = request.POST['coordinates']
            coordinates = coordinates.split('\n')
            x = []
            y = []
            line_num = 1
        
            for i in coordinates:
                line = i.split(' ')
                if(len(line) != 2):
                    response = JsonResponse({'Fail':'Number of tokens on line {} are not equal to 2'.format(line_num)})
                    response.status_code = 403
                    return response
                try:
                    _x_ = float(line[0])
                    _y_ = float(line[1])
                    x.append(_x_)
                    y.append(_y_)
                except ValueError:
                     response = JsonResponse({'Fail':'Invalid entry on line {}'.format(line_num)})
                     response.status_code = 403
                     return response
                line_num += 1   
            
            if(len(set(coordinates)) != n + m or len(coordinates) != n + m):
                response = JsonResponse({'Fail':'Number of Distinct Coordinates are not equal to {}'.format(n + m)})
                response.status_code = 403
                return response
            Model.execute(
                    customer_nodes = n, 
                    depot_nodes = m,
                    max_vehicles = max_vehicles,
                    iterations = iterations,
                    bucket_size = bucket_size,
                    max_no_improve = max_no_improve,
                    x_coordinates = x,
                    y_coordinates = y
            )  
               
        except:
            print('Internal Error')    
        
        
        response = {
                'Success':'Your form has been submitted successfully' # response message
        }
       
        return JsonResponse(response)