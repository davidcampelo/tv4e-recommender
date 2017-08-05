from django.shortcuts import render
from django.http import HttpResponse
from tv4e.models import Asgie, InformationSources

def index(request):
	asgies = Asgie.objects.all()
	context = {'asgies': asgies}
	return render(request, 'videos/index.html', context)

def video_detail(request, video_id):
	return HttpResponse("You're looking at video %s." % video_id)

def asgie_detail(request, asgie_id):
	asgie = Asgie.objects.get(id=asgie_id)
	information_sources = InformationSources.objects.filter(asgie=asgie)
	context = {'asgie': asgie, 'information_sources': information_sources}
	return render(request, 'videos/asgie_detail.html', context)
