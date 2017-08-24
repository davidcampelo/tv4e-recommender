# encoding: utf-8
import logging
import random
import django
import decimal

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




user_id = Senior.objects.get(pk=3).id # XXX h0h0h0
contents = InformativeVideos.objects.all()

content_set = set()

while len(content_set) < 20:
    random_content_index = random.randint(0, len(contents) - 1)
    content_id = contents[random_content_index].id
    content_set.add(content_id)
    
    rating = Rating(user_id=user_id, content_id=content_id, rating=round(decimal.Decimal(random.uniform(0, 1)), 1))
    rating.save()
    print rating


