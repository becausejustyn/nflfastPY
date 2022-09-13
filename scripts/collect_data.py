
### Betting

import urllib2
from bs4 import BeautifulSoup
import re
import pandas as pd
import sys
import os.path

### Collect Betting Line Data

base_url = "https://www.teamrankings.com/nfl/odds-history/results/"
output_location = os.path.join(sys.argv[1], sys.argv[2])
opener = urllib2.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
page = BeautifulSoup(opener.open(base_url), 'html.parser')
table_data = page.find_all("tr", {"class": "text-right nowrap"})
betting_lines = []
for line in table_data:
    line_list = str(line).splitlines()
    try:
        betting_lines.append([re.search('<td>(.*)</td>', line_list[1]).group(1),
                              line_list[4].split(">")[1].split("<")[0]])
    except:
        betting_lines.append([None, None])

historic_lines_df = pd.DataFrame(betting_lines)
historic_lines_df.columns = ['spread', 'win_pct']
historic_lines_df.to_csv(output_location, index = False)

### Collect Game Data

from urllib import urlopen

base_url = 'http://www.footballdb.com/teams/nfl/new-england-patriots/teamvsteam?opp='
game_data = []
n_teams = 32
output_location = os.path.join(sys.argv[1], sys.argv[2])
for team_number in range(1, n_teams + 1, 1):
    page  = str(BeautifulSoup(urlopen(base_url + str(team_number)), 
                              'html.parser').findAll("table"))
    for row in [x.split("<td>") for x in page.split("row")]:
        try:
            game_date, outcome = str(re.findall('gid=(.*)', row[4])).split(">")[:2]
            game_data.append([game_date[2:10], outcome[0]])
        except:
            continue
game_data_df = pd.DataFrame(game_data)
game_data_df.columns = ['date', 'outcome']
game_data_df.to_csv(output_location,  index = False)
