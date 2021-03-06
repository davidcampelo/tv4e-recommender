# +TV4E recommender system

The [+TV4E Project](http://www.tv4e.pt/)[1] is an interactive television platform which allows automatically the enrichment of television experience with the integration of social contents. In order to achieve a more compelling and personalized approach a system for recommending video contents according to the users' preferences.

This system has been developed in the scope of a PhD research [2] in the context of the Doctoral Program of Information and  Communication in Digital Platforms (University of Aveiro, Portugal). It proposes a context aware recommender system (CARS) of informative contents about Assistance  Services  of  General  Interest  for the Elderly (ASGIE), for later exhibition on an Interactive TV (iTV) platform. The motivation of this research is to enhance the TV watching experience and promote seniors’ autonomy, wellbeing and info-inclusion by providing personalized high-valued informative contents. 

An example of installation of this code was installed in a web server for tests with potential end users and can be found at [http://79.137.39.168:8080](http://79.137.39.168:8080/view/).

## Installation/Configuration guide:

System requirements:
* Ubuntu 16.04
* Redis DB (Click [here](https://hostpresto.com/community/tutorials/how-to-install-and-configure-redis-on-ubuntu-14-04/)  for instructions)
* Python 3.5.x
* Python3-TK library (sudo apt-get install python3-tk)
* Python MySQL integration library (sudo apt-get install libmysqlclient-dev)

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
> python manage.py runserver 127.0.0.1:8001
```
Running the server like this, will make the website available 
[http://127.0.0.1:8001](http://127.0.0.1:8001) other applications also use this port
so you might need to try out 8002 instead. 

### Closing down.
* when you are finished running the project you can exit the virtual env:
```bash
> deactivate
```
## References
Please cite +TV4E Project if it helps your research. You can use the following BibTeX entries:
```
[1]
@inproceedings{Campelo:2017:RPI:3084289.3084292,
	 author = {Campelo, David and Silva, Telmo and Ferraz de Abreu, Jorge},
	 title = {Recommending Personalized Informative Contents on iTV},
	 booktitle = {Adjunct Publication of the 2017 ACM International Conference on Interactive Experiences for TV and Online Video},
	 series = {TVX '17 Adjunct},
	 year = {2017},
	 isbn = {978-1-4503-5023-5},
	 location = {Hilversum, The Netherlands},
	 pages = {99--103},
	 numpages = {5},
	 url = {http://doi.acm.org/10.1145/3084289.3084292},
	 doi = {10.1145/3084289.3084292},
	 acmid = {3084292},
	 publisher = {ACM},
	 address = {New York, NY, USA},
	 keywords = {context-aware, elderly, info-inclusion, interactive tv, recommender systems},
}
```
