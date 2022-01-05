from datetime import datetime
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

# Scrape for links in website database given the queries
for query in desired_queries:
    for i in range(3):
        # 99):
        page = page+i
        
        # Constracting http query
        url = 'https://www.efsyn.gr/search?keywords=' + quote(query) + '&page=' + str(page)
            
        # To avoid 403-error using User-Agent
        req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"})
        response = urllib.request.urlopen( req )
        
        # Check page accessibility
        if response.status_code == 200:
            
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

# Scrape inside individual search results
for i in range(len(link_results)):
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
        article_results.append(i.text)


# Add articles to pandas dataframe
data = {'Article': article_results, 'Date': datetime.now()}
dataset = pd.DataFrame(data=data)
cols = ['Article', 'Date']
dataset = dataset[cols]

# A couple of development sanity checks:
print(link_results)
print(len(link_results))

# Check if link_results are unique:
if len(link_results) > len(set(link_results)):
   print("not unique")
else:
    print("unique") 

print(dataset)