import sys
import json
import datetime
import zlib
import msgpack
import time

sys.path.insert(0, "src")
import helper_functions as hf
import io_helper as ioh
import datetime_helper as dh

sys.path.insert(1, 'src')
# Include config and class
from config import key, secret
from Class import ApiAccess

# Save location
path_save_data = "data"
filename_article_data = "{:s}/article_data.json".format(path_save_data)

# Declare your key and secret
obj = ApiAccess(key, secret)

# Declare endpoint
endpoint = 'public/trending-news-data'

# No parameters required
datetime_start = datetime.datetime(2019, 6, 1)
datetime_end = datetime.datetime(2020, 5, 30)

date = datetime_start;

# Cryptocurrency symbol to keyword map

sym_to_keyword = {
	'XBTUSD':'bitcoin',
	'ETHUSD':'ethereum',
	'XRPUSD':'ripple',
}

# USER SETTING (Use one of the above mapped symbols)
symbol = 'XBTUSD'

article_data = {"symbol":symbol,"data":[]};


while(date <= datetime_end):

	payload = {"date":date.strftime("%Y-%m-%d"),'keyword_contains':sym_to_keyword[symbol]};

	time.sleep(0.2);

	brief_data = [];

	result = obj.send(endpoint, payload, dict());

	n = len(result['returned']['data']);

	for i in range(n):
		brief_data.append(result['returned']['data'][i]['brief']);

	article_data["data"].append({"brief_data":brief_data,"date":date.strftime("%Y-%m-%d")});

	print("Got data from"+date.strftime("%Y-%m-%d"));

	# print(json.dumps(article_data,indent = 4))

	date += datetime.timedelta(days=1);


# print(json.dumps(article_data, indent=4));

# check if the data path exists
ioh.check_path(path_save_data, create_if_not_exist=True)

# save the data
print("saving data to {:s}".format(filename_article_data))
with open(filename_article_data, "w") as f:
	json.dump(article_data,f)

print("done!")

