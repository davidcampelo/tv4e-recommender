# encoding: utf-8
import logging
import random
import django
import decimal
from django.utils import timezone

# XXX work around to run this script in a sub-directory of the project (/scripts)
import os
import sys
root_path = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(root_path, '../'))
sys.path.insert(0, root_path)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tv4e.settings")
django.setup()


from view.models import Asgie, InformativeVideos, AsgieAvResource, Boxes
from rs.models import Senior, Rating


seniors = Senior.objects.all().values('id')
contents = InformativeVideos.objects.all()
for senior in seniors:
	user_id = senior['id']
	content_set = set()
	while len(content_set) < 5:
	    random_content_index = random.randint(0, len(contents) - 1)
	    content_id = contents[random_content_index].id
	    content_set.add(content_id)
	    
	    timestamp = timezone.now()
	    timestamp = timestamp.replace(day=random.randint(1,timestamp.day))

	    rating = Rating(user_id=user_id, 
	    	content_id=content_id, 
	    	rating=round(decimal.Decimal(random.uniform(-1, 0)), 1),
	    	rating_timestamp=timestamp)
	    rating.save()
	    print rating


