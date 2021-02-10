import sys
import json
import datetime
import time
import re, string
import msgpack
import zlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.animation as animation

from textblob import TextBlob

import ema

sys.path.insert(0, "src")
import io_helper as ioh

def clean_article(article):

    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", article).split())

def get_article_sentiment(article):
        
    analysis = TextBlob(clean_article(article))

    # Removing 'bitcoin cash' articles from the data
    if(re.search("([Bb]itcoin\s[Cc]ash)",article)):
        return 'neutral'

    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'



# define the location of the price data file
filename_bitmex_data = "data/bitmex_data.msgpack.zlib"

# load the price data
with open(filename_bitmex_data, "rb") as f:
	temp = msgpack.unpackb(zlib.decompress(f.read()))
	price_symbol = temp[0]['symbol']
	t_price_data = np.array([el["t_epoch"] for el in temp], dtype=np.float64)
	#price_data = np.array([el["open"] for el in temp], dtype=np.float64)
	price_data = np.array([el["close"] for el in temp], dtype=np.float64)

# initialise some labels for the plot
datenum_price_data = [md.date2num(datetime.datetime.fromtimestamp(el)) for el in t_price_data]

datenum_price_data = datenum_price_data[::24]
price_data = price_data[::24]

# define the location of the input file
filename_article_data = "data/article_data.json"

# load the article data
with open(filename_article_data, "rb") as f:
	article_data = json.load(f);
	article_symbol = article_data['symbol']

# check if price and article cryptocurrencies match

if(article_symbol != price_symbol):
	print("Price Symbol:",price_symbol)
    print("Article Symbol:",article_symbol)
    sys.exit("Price and Article symbols don't match! Please match the symbols and re-run loading scripts.")


positive_data = [];
negative_data = [];
neutral_data = [];

ratio = []

days = len(article_data['data']);

for i in range(days):
    articles = len(article_data['data'][i]['brief_data']);
    day_sentiment = [];
    bull = 0;
    # Initial value is 1 to avoid zero division
    bear = 1;
    neut = 0;
    total = 1;
    for j in range(articles):

        day_sentiment.append(get_article_sentiment(article_data['data'][i]['brief_data'][j]));

        if(day_sentiment[-1] == 'positive'):
            bull+=1;
        elif(day_sentiment[-1] == 'negative'):
            bear+=1;
        else:
            neut+=1;

        total += 1;


    positive_data.append(bull/total);
    negative_data.append(bear/total);
    neutral_data.append(neut/total);
    ratio.append(bull/bear)




# Simulation to find the best moving average windows

best_window = [-1000,2,3]
best = [-1000,0,0];

window = 2;
while(window<=10):

    mvngavg_ratio = ema.ewma_vectorized_safe(ratio, 2/(window+1))

    window2 = 3;
    while(window2 <= 25):
        ratio_signal = [];

        for i in range(window2-1):
        	x = mvngavg_ratio[0:window2]
        	ratio_signal.append((x[-1] - np.mean(x))/np.std(x));

        for i in range(window2-1,len(mvngavg_ratio)):
            x = mvngavg_ratio[i-window2+1:i+1];
            ratio_signal.append((x[-1] - np.mean(x))/np.std(x));

        # Simulation to find the bias in articles

        pnl = [0]*len(ratio_signal);
        pnl[0] = 1.0;

        p = 0.0;

        scenario = [];

        while(p<=0.6):
            n=0.0;
            while(n<=0.6):
                pnl = [0]*len(ratio_signal);
                pnl[0] = 1.0;
                for i in range(1,len(ratio_signal)):
                    if ratio_signal[i-1] > p:
                        pnl[i] = (price_data[i] / price_data[i-1]) * pnl[i-1];
                    elif ratio_signal[i-1] < -1*n:
                        pnl[i] = (price_data[i-1] / price_data[i]) * pnl[i-1];
                    else:
                        pnl[i] = pnl[i-1]
                if(best[0]<pnl[-1]):
                    best[0]=pnl[-1];
                    best[1]=p;
                    best[2]=n;
                scenario.append([p,n,pnl[-1]]);

                n+=0.1;
            p+=0.1;

        # End of bias simulation

        if(best[0] > best_window[0]):
            best_window[0] = best[0];
            best_window[1] = window;
            best_window[2] = window2;

        window2+=1;

    window+=1;

# End of best window simulation

print("Best profit:",round(best[0],3));
print("Positive bias:",round(best[1],3));
print("Negative bias: ",round(best[2],3));
print("Best Exponential Moving Average window: ",best_window[1]);
print("Best Signal window: ",best_window[2]);

# Moving average windows set using simulations
window = best_window[1]
window2 = best_window[2];



# Moving average calculation using the best windows

mvngavg_ratio = [];

# Creates EMA of 'ratio' with 'window'
mvngavg_ratio = ema.ewma_vectorized_safe(ratio, 2/(window+1))

ratio_signal = [];

for i in range(window2-1):
	x = mvngavg_ratio[0:window2]
	ratio_signal.append((x[-1] - np.mean(x))/np.std(x));

for i in range(window2-1,len(mvngavg_ratio)):
    x = mvngavg_ratio[i-window2+1:i+1];
    ratio_signal.append((x[-1] - np.mean(x))/np.std(x));


# Profit and Loss calculation

pnl = [0]*len(ratio_signal);
pnl[0] = 1.0;

for i in range(1,len(ratio_signal)):

    if ratio_signal[i-1] > best[1]:
        pnl[i] = (price_data[i] / price_data[i-1]) * pnl[i-1];
    elif ratio_signal[i-1] < -1*best[2]:
        pnl[i] = (price_data[i-1] / price_data[i]) * pnl[i-1];
    else:
        pnl[i] = pnl[i-1]


# Creating JSON output

article_results = {'symbol':price_symbol,'data':[]}
for i in range(len(datenum_price_data)):
	day = {
		'date':md.num2date(datenum_price_data[i]).strftime("%Y-%m-%dT%H:%M:%SZ"),
		'bbratio':ratio[i],
		'mvngavg_ratio':mvngavg_ratio[i],
		'ratio_signal':ratio_signal[i],
		'wallet':pnl[i],
	}
	article_results['data'].append(day)

# print(json.dumps(final_data, indent=4))



# Saving the final json

# Save location
path_save_data = "results"
filename_article_results = "{:s}/article_results.json".format(path_save_data)

# check if the data path exists
ioh.check_path(path_save_data, create_if_not_exist=True)

# save the data
print("saving data to {:s}".format(filename_article_results))
with open(filename_article_results, "w") as f:
	json.dump(article_results,f)

print("Saved!")


# set up the figure
fig, ax = plt.subplots(4, 1, sharex=True, sharey=False)

# plot stuff
ax[0].grid(linewidth=0.4)
ax[1].grid(linewidth=0.4)
ax[2].grid(linewidth=0.4)
ax[3].grid(linewidth=0.4)

ax[0].plot(datenum_price_data, price_data, linewidth=0.5)

# def animate(i):

ax[1].plot(datenum_price_data, ratio, color="b", linewidth=0.5)
ax[1].plot(datenum_price_data, mvngavg_ratio, color="r", linewidth=0.5)

ax[2].plot(datenum_price_data, ratio_signal, color="r", linewidth=0.5)

ax[3].plot(datenum_price_data, pnl, color="r", linewidth=0.5)

# label axes
ax[0].set_ylabel("Price")
ax[1].set_ylabel("Ratio")
ax[2].set_ylabel("Trading Signal")
ax[3].set_ylabel("Wallet")

# legend
ax[1].legend(["Ratio","EMA"])

# generate the time axes
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax[0]=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
ax[0].xaxis.set_major_formatter(xfmt)

plt.gcf().set_size_inches(32, 18)
# save the plot
plt.savefig('plots/article_sentiment.png', bbox_inches='tight')

# show the plot
# ani = animation.FuncAnimation(fig, animate, frames=days, interval=1) 
plt.show()


