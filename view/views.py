from django.shortcuts import render
from django.http import HttpResponse
from django.views import generic
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

from django.db.models import Avg

from majordomo.models import Asgie, Video, User, Rating

# VIDEOS
###################################################################################################################
def video_index(request):

    paginate_by = 8

    # asgie_selected = request.GET.get('asgie')
    # if asgie_selected:
    #     selected = Asgie.objects.filter(id=asgie_selected)[0]
    #     videos = selected.movies.order_by('-created_at')
    # else:
    videos = Video.objects.order_by('-id')
    videos_count = len(videos)
    asgies = Asgie.objects.all()

    paginator = Paginator(videos, paginate_by)

    page_number = request.GET.get("page")

    try:
        page = paginator.page(page_number)
    except PageNotAnInteger:
        page_number = 1
        page = paginator.page(page_number)
    except EmptyPage:
        page = paginator.page(videos.count())

    page_number = int(page_number)
    page_start = 1 if page_number < 5 else page_number - 3
    page_end = 6 if page_number < 5 else page_number + 2

    context_dict = {'videos': page,
                    'asgies': asgies,
                    'videos_count': videos_count,
                    'pages': range(page_start, page_end),
                    }

    return render(request, 'view/video_index.html', context_dict)




def video_detail(request, video_id):
    video = Video.objects.get(id=video_id)
    # videotokens = VideoTokens.objects.get(pk=video_id)
    context = {'video': video}
    return render(request, 'view/video_detail.html', context)


# ASGIE
###################################################################################################################
class AsgieIndexView(generic.ListView):
    template_name = 'view/asgie_index.html'
    context_object_name = 'asgies'

    def get_queryset(self):
        return Asgie.objects.all()


def asgie_detail(request, asgie_id):
    asgie = Asgie.objects.get(id=asgie_id)
    context = {'asgie': asgie}
    return render(request, 'view/asgie_detail.html', context)


# USERS
###################################################################################################################
class UserIndexView(generic.ListView):
    template_name = 'view/user_index.html'
    context_object_name = 'items'

    def get_queryset(self):
        return User.objects.all()


def user_detail(request, user_id):
    item = User.objects.get(pk=user_id)
    ratings = Rating.objects.filter(user_id=user_id).order_by('date_creation')
    context = {'item': item, 'ratings': ratings} 
    if len(ratings) > 0:
        mean = ratings.aggregate(value=Avg('overall_rating_value'))
        context['mean'] = round(mean['value'],3)
    
    return render(request, 'view/user_detail.html', context)


# ANALYTICS
###################################################################################################################

def analytics(request):
    return render(request, 'view/analytics.html', {})