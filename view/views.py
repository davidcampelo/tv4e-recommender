from django.shortcuts import render
from django.http import HttpResponse
from django.views import generic

from tv4e import models



class VideoIndexView(generic.ListView):
    template_name = 'view/video_index.html'
    context_object_name = 'videos'

    def get_queryset(self):
        """Return the last five published questions."""
        return models.InformativeVideos.objects.all()



def video_detail(request, video_id):
	pass


class AsgieIndexView(generic.ListView):
    template_name = 'view/asgie_index.html'
    context_object_name = 'asgies'

    def get_queryset(self):
        """Return the last five published questions."""
        return models.Asgie.objects.all()

def asgie_detail(request, asgie_id):
	asgie = models.Asgie.objects.get(id=asgie_id)
	resources = [aar.av_resource.url for aar in models.AsgieAvResource.objects.filter(asgie=asgie)]
	# information_sources = InformationSources.objects.filter(asgie=asgie)
	context = {'asgie': asgie, 'resources': resources}
	return render(request, 'view/asgie_detail.html', context)
