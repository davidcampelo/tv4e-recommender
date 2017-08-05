from django.shortcuts import render
from django.http import HttpResponse
from tv4e.models import Asgie, InformationSources, AvResources, AsgieAvResource

def index(request):
	asgies = Asgie.objects.all()
	context = {'asgies': asgies}
	return render(request, 'videos/index.html', context)

def asgie_index(request):
	asgies = Asgie.objects.all()
	context = {'asgies': asgies}
	return render(request, 'videos/asgie_index.html', context)

def asgie_detail(request, asgie_id):
	asgie = Asgie.objects.get(id=asgie_id)
	resources = [aar.av_resource.url for aar in AsgieAvResource.objects.filter(asgie=asgie)]
	# information_sources = InformationSources.objects.filter(asgie=asgie)
	context = {'asgie': asgie, 'resources': resources}
	return render(request, 'videos/asgie_detail.html', context)
