# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
#
# Also note: You'll have to insert the output of 'django-admin.py sqlcustom [app_label]'
# into your database.
from __future__ import unicode_literals

from django.db import models


class Asgie(models.Model):
    title = models.CharField(unique=True, max_length=250)
    title_pt = models.CharField(max_length=100)
    image = models.CharField(max_length=200)
    color = models.CharField(max_length=100)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    deleted_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'asgie'

    def __str__(self):
        return '[{}] {}'.format(self.id, self.title)

# ALTER TABLE `tv4e`.`asgie_av_resource` 
# ADD COLUMN `id` INT(10) NOT NULL FIRST,
# DROP PRIMARY KEY;

class AsgieAvResource(models.Model):
    asgie = models.ForeignKey(Asgie)
    av_resource = models.ForeignKey('AvResources')
    selected = models.IntegerField()
    duration_percentage = models.FloatField(blank=True, null=True)
    deleted_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'asgie_av_resource'


class AvResourceInformativeVideo(models.Model):
    av_resource = models.ForeignKey('AvResources')
    informative_video = models.ForeignKey('InformativeVideos')
    deleted_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'av_resource_informative_video'


class AvResources(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    url = models.CharField(unique=True, max_length=255)
    name = models.CharField(max_length=250, blank=True)
    av_resources_type = models.ForeignKey('AvResourcesTypes')
    deleted_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'av_resources'


class AvResourcesTypes(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    type = models.CharField(unique=True, max_length=100)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    deleted_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'av_resources_types'


class BoxInformativeVideo(models.Model):
    box = models.ForeignKey('Boxes')
    informative_video = models.ForeignKey('InformativeVideos')
    seen = models.IntegerField()
    rejected = models.IntegerField()
    deleted_at = models.DateTimeField(blank=True, null=True)
    sent_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'box_informative_video'


class Boxes(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    uu_id = models.IntegerField()
    serial = models.CharField(max_length=255)
    on_state = models.IntegerField()
    city = models.ForeignKey('Cities')
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    deleted_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'boxes'


class CartaSocial(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    resposta = models.CharField(max_length=250)

    class Meta:
        managed = False
        db_table = 'carta_social'


class Cities(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    name = models.CharField(max_length=200)
    district = models.ForeignKey('Districts')
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    deleted_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cities'


class Districts(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    name = models.CharField(unique=True, max_length=100)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    deleted_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'districts'


class Filters(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    filter = models.CharField(max_length=255)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    deleted_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'filters'


class InformationSources(models.Model):
    url = models.CharField(unique=True, max_length=255)
    news_container = models.CharField(max_length=500)
    news_link = models.CharField(max_length=500)
    content_container = models.CharField(max_length=500)
    content_title = models.CharField(max_length=500)
    content_description = models.CharField(max_length=500)
    html_exception = models.CharField(max_length=1000, blank=True)
    asgie = models.ForeignKey(Asgie, related_name="information_sources")
    city = models.ForeignKey(Cities, blank=True, null=True)
    deleted_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'information_sources'


class InformationSourcesSubs(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    url = models.CharField(max_length=1000)
    city = models.ForeignKey(Cities, blank=True, null=True)
    information_source = models.ForeignKey(InformationSources)
    content = models.IntegerField(blank=True, null=True)
    carta_social_id = models.IntegerField(blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    deleted_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'information_sources_subs'


class InformativeVideos(models.Model):
    desc = models.CharField(max_length=10000)
    title = models.CharField(max_length=500)
    information_source = models.ForeignKey(InformationSources, blank=True, null=True)
    information_sources_sub = models.ForeignKey(InformationSourcesSubs, blank=True, null=True)
    av_resource = models.ForeignKey(AvResources)
    duration = models.IntegerField(blank=True, null=True)
    deleted_at = models.DateTimeField(blank=True, null=True)
    expired_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'informative_videos'

    @property
    def asgie_title_pt(self):
        source = self.information_source
        if source is None:
            source = self.information_sources_sub
            source = source.information_source
        return source.asgie.title_pt

    def __str__(self):
        return self.id


class Logs(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    on_state = models.IntegerField(blank=True, null=True)
    event = models.CharField(max_length=200, blank=True)
    informative_video = models.ForeignKey(InformativeVideos, blank=True, null=True)
    box = models.ForeignKey(Boxes)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'logs'


class Migrations(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    migration = models.CharField(max_length=255)
    batch = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'migrations'


class OauthAccessTokens(models.Model):
    id = models.CharField(primary_key=True, max_length=100)
    user_id = models.IntegerField(blank=True, null=True)
    client_id = models.IntegerField()
    name = models.CharField(max_length=255, blank=True)
    scopes = models.TextField(blank=True)
    revoked = models.IntegerField()
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    expires_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'oauth_access_tokens'


class OauthAuthCodes(models.Model):
    id = models.CharField(primary_key=True, max_length=100)
    user_id = models.IntegerField()
    client_id = models.IntegerField()
    scopes = models.TextField(blank=True)
    revoked = models.IntegerField()
    expires_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'oauth_auth_codes'


class OauthClients(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    user_id = models.IntegerField(blank=True, null=True)
    name = models.CharField(max_length=255)
    secret = models.CharField(max_length=100)
    redirect = models.TextField()
    personal_access_client = models.IntegerField()
    password_client = models.IntegerField()
    revoked = models.IntegerField()
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'oauth_clients'


class OauthPersonalAccessClients(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    client_id = models.IntegerField()
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'oauth_personal_access_clients'


class OauthRefreshTokens(models.Model):
    id = models.CharField(primary_key=True, max_length=100)
    access_token_id = models.CharField(max_length=100)
    revoked = models.IntegerField()
    expires_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'oauth_refresh_tokens'


class PasswordResets(models.Model):
    email = models.CharField(max_length=255)
    token = models.CharField(max_length=255)
    created_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'password_resets'


class Scores(models.Model):
    box = models.ForeignKey(Boxes)
    asgie = models.ForeignKey(Asgie)
    score = models.IntegerField()
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    deleted_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'scores'


class Users(models.Model):
    id = models.IntegerField(primary_key=True)  # AutoField?
    name = models.CharField(max_length=255)
    email = models.CharField(unique=True, max_length=255)
    password = models.CharField(max_length=255)
    remember_token = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'users'
