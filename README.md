# +TV4E recommender system

The [+TV4E Project](http://socialitv.web.ua.pt/index.php/projects/sponsored-projects/tv4e/) is an interactive television platform which allows automatically the enrichment of television experience with the integration of social contents. In order to achieve a more compelling and personalized approach a system for recommending video contents according to the users' preferences.

This systsem is still under  development, and so this README is still under construction.

# Installation guide:

System requirements:
* Ubuntu 14.04
* Redis DB (Click [here](https://hostpresto.com/community/tutorials/how-to-install-and-configure-redis-on-ubuntu-14-04/)  for instructions)
* Python 2.7.x

For windows users it's a good idea to install the Anaconda package. Anaconda is the leading open data science platform powered by Python (according to their homepage) [Anaconda](https://www.continuum.io/downloads)
 
### Download code
```bash
> git clone https://github.com/davidcampelo/tv4e-recommender
```
### Create a virtual environment for the project 
Look at the following guide for more details [guide](http://docs.python-guide.org/en/latest/dev/virtualenvs/#virtualenvironments-ref)
 
```bash
> cd tv4e-recommender
> virtualenv tv4e_project -p python-2.7
```
To start/open your development environment:
```
> source tv4e_project/bin/activate
```
To close your development environment:
```
> deactivate
```


if you are running Anaconda you can also use conda virtual environment instead.
### Get the required packages

```bash
pip install -r requirements.txt
```

### Configure REDIS
First, make sure you have a local redis instance running. The engine expects to find redis at redis://localhost:6379. For any different configuration please check the tv4e/settings.py file.

### Create the DB migrations 
If you have a database running on your machine I would encourage 
you to connect it, by updating the settings in `tv4_recommender/settings.py` 

To set up another database is described in the Django docs [here](https://docs.djangoproject.com/en/1.10/ref/databases/)
```bash
> python manage.py makemigrations
> python manage.py migrate
```

### Start the web server
 To start the development server run:
```bash
> python manage.py runserver 127.0.0.1:8000
```
Running the server like this, will make the website available 
[http://127.0.0.1:8001](http://127.0.0.1:8001) other applications also use this port
so you might need to try out 8002 instead. 

### Closing down.
* when you are finished running the project you can exit the virtual env:
```bash
> deactivate
```
