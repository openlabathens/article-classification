from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote

# List with queries I want to make
desired_queries = ['γυναικοκτονία']
# , 'δολοφονία', 'σύζυγος']

results = []
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
        
        # Extracting number of results
        search = soup.find_all('div', attrs={'class':'default-teaser triple'})
        
        # Search for articles within given div:
        for s in  search:
            articles = soup.find_all('article', attrs={'class': 'default-teaser__article default-teaser__article'})
            
            # Extract the link of each article:
            for a in articles:
                links = "https://www.efsyn.gr" + a.contents[9].get('href')
                # print(links)  
                results.append(links)

print(results)
print(len(results))

# check if results are unique:
if len(results) > len(set(results)):
   print("not unique")
else:
    print("unique")

for i in range(len(results)):
    # print(results[i])
    # TODO: take each link from results and parse it to get the raw text.
    url = results[i]
    
    # To avoid 403-error using User-Agent
    req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"})
    response = urllib.request.urlopen( req )
    
    html = response.read()
    
    # Parsing response
    soup = BeautifulSoup(html, 'html.parser')

    body = soup.find('div', attrs={'class':'article__body'})
    
    for i in body:
        # TODO: add articles to pandas dataframe
        print(i.text)