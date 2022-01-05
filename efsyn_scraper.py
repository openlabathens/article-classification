from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote

import pandas as pd
import numpy as np

# List with queries I want to make
desired_queries = ['γυναικοκτονία']
# , 'δολοφονία', 'σύζυγος']

link_results = []
article_results = []
page = 0

for query in desired_queries:
    for i in range(3):
        # 99):
        page = page+i
        # Constracting http query
        url = 'https://www.efsyn.gr/search?keywords=' + quote(query) + '&page=' + str(page)
            
        # To avoid 403-error using User-Agent
        req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"})
        response = urllib.request.urlopen( req )
        
        html = response.read()
        
        # Parsing response
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extracting number of link_results
        search = soup.find_all('div', attrs={'class':'default-teaser triple'})
        
        # Search for articles within given tag:
        for s in  search:
            articles = soup.find_all('article', attrs={'class': 'default-teaser__article default-teaser__article'})
            
            # Extract the link of each article:
            for a in articles:
                links = "https://www.efsyn.gr" + a.contents[9].get('href')
                # print(links)  
                link_results.append(links)

for i in range(len(link_results)):
    # print(link_results[i])
    # TODO: take each link from link_results and parse it to get the raw text.
    url = link_results[i]
    
    # To avoid 403-error using User-Agent
    req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"})
    response = urllib.request.urlopen( req )
    
    html = response.read()
    
    # Parsing response
    soup = BeautifulSoup(html, 'html.parser')

    # Get article body
    body = soup.find('div', attrs={'class':'article__body'})
    
    

    for i in body:
        # TODO: add articles to pandas dataframe
        print(i.text)
        article_results.append(i.text)

# A couple of development sanity checks:
print(link_results)
print(len(link_results))

# Check if link_results are unique:
if len(link_results) > len(set(link_results)):
   print("not unique")
else:
    print("unique") 