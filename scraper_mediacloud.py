import mediacloud.api, json, datetime
from dotenv import load_dotenv
import os 

load_dotenv()

api_key = os.environ.get('MY_API_KEY')

mc = mediacloud.api.MediaCloud(api_key)
stories = []

last_processed_stories_id = 0
fetched_stories = mc.storyList('γυναικοκτονία OR δολοφονία OR γυναίκα AND tags_id_media:34412477', 
                                   solr_filter=mc.dates_as_query_clause(datetime.date(2020,1,1), datetime.date(2021,12,31)))
stories.extend(fetched_stories)
print(json.dumps(stories))

res = mc.storyCount('γυναικοκτονία OR δολοφονία OR γυναίκα AND tags_id_media:34412477', 'publish_date:[NOW-1YEAR TO NOW]')
print(res['count']) # prints the number of stories found