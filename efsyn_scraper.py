from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote

# List with queries I want to make
desired_queries = ['γυναικοκτονία']
# , 'δολοφονία', 'σύζυγος']

results = []
page = 0

