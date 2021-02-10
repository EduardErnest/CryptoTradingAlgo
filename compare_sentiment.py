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
filename_article_results = "results/article_results.json"

# load the article data
with open(filename_article_results, "rb") as f:
	article_results = json.load(f);
	article_symbol = article_results['symbol']

# define the location of the input file
filename_funding_results = "results/funding_results.json"

# load the article data
with open(filename_funding_results, "rb") as f:
    funding_results = json.load(f);
    funding_symbol = funding_results['symbol']

# define the location of the input file
filename_combined_results = "results/combined_results.json"

# load the article data
with open(filename_combined_results, "rb") as f:
    combined_results = json.load(f);
    combined_symbol = combined_results['symbol']

# check if price and funding cryptocurrencies match

if( not(article_symbol == funding_symbol == combined_symbol) ):
    print("Article Symbol:",article_symbol)
    print("Funding Symbol:",funding_symbol)
    print("Combined Symbol:",combined_symbol)
    sys.exit("Funding, Article, and Combined symbols don't match! Please match the symbols and re-run loading scripts.")




# Loading the article signal

article_signal = []
article_wallet = []
for i in range(len(article_results['data'])):
    article_signal.append(article_results['data'][i]['ratio_signal'])
    article_wallet.append(article_results['data'][i]['wallet'])



# Loading the funding signal

funding_signal = []
funding_wallet = []
for i in range(len(funding_results['data'])):
    funding_signal.append(funding_results['data'][i]['funding_signal'])
    funding_wallet.append(funding_results['data'][i]['wallet'])

temp = funding_signal[-1]
funding_signal = funding_signal[::3]

# Last date cut off during slicing is attached again
funding_signal.append(temp)


temp = funding_wallet[-1]
funding_wallet = funding_wallet[::3]

# Last date cut off during slicing is attached again
funding_wallet.append(temp)



# Loading the combined signal

combined_signal = []
combined_wallet = []
for i in range(len(combined_results['data'])):
    combined_signal.append(combined_results['data'][i]['combined_signal'])
    combined_wallet.append(combined_results['data'][i]['wallet'])





# set up the figure
fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)

# plot stuff
ax[0].grid(linewidth=0.4)
ax[1].grid(linewidth=0.4)
# ax[2].grid(linewidth=0.4)
# ax[3].grid(linewidth=0.4)

ax[0].plot(datenum_price_data, price_data, linewidth=0.5)

# def animate(i):

ax[1].plot(datenum_price_data, article_wallet, color="b", linewidth=0.5)

ax[1].plot(datenum_price_data, funding_wallet, color="r", linewidth=0.5)

ax[1].plot(datenum_price_data, combined_wallet, color="g", linewidth=0.5)


# label axes
ax[0].set_ylabel("Price")
ax[1].set_ylabel("Wallet")
# ax[2].set_ylabel("Combined Signal")
# ax[3].set_ylabel("Wallet")

# legend
ax[1].legend(["Article","Funding","Combined"])

# generate the time axes
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax[0]=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
ax[0].xaxis.set_major_formatter(xfmt)

plt.gcf().set_size_inches(32, 18)

# save the plot
plt.savefig('plots/compare_sentiment.png', bbox_inches='tight')

# show the plot
# ani = animation.FuncAnimation(fig, animate, frames=days, interval=1) 
plt.show()


