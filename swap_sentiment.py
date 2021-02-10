import sys
import json
import datetime
import time
import re, string
import msgpack
import zlib
import numpy as np
import matplotlib
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




# define the location of the funding data file
filename_funding_data = "data/funding_data.msgpack.zlib"

# load the funding data
with open(filename_funding_data, "rb") as f:
	temp = msgpack.unpackb(zlib.decompress(f.read()))
	funding_symbol = temp[0]['symbol']
	t_funding_data = np.array([el["t_epoch"] for el in temp], dtype=np.float64)
	#price_data = np.array([el["open"] for el in temp], dtype=np.float64)
	funding_data = np.array([el["fundingRate"] for el in temp], dtype=np.float64)

# initialise some labels for the plot
datenum_funding_data = [md.date2num(datetime.datetime.fromtimestamp(el)) for el in t_funding_data]

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

offset = datenum_price_data.index(datenum_funding_data[0])

# Since we get price data every 1hr we take closing prices every 8th hour
datenum_price_data = datenum_price_data[offset::8]
price_data = price_data[offset::8]

# check if price and funding rate cryptocurrencies match

if(funding_symbol != price_symbol):
	print("Price Symbol:",price_symbol)
    print("Funding Symbol:",funding_symbol)
	sys.exit("Price and Funding symbols don't match! Please match the symbols and re-run loading scripts.")

# Simulation to find the best EMA window

best_window = 2
best = -1000;
reversal = [-1000,0.001,0.001]
days = len(datenum_price_data)

window = 2;
while(window<=50):

	mvngavg_funding = ema.ewma_vectorized_safe(funding_data, 2/(window+1))

	funding_signal = [];

	for i in range(len(mvngavg_funding)):
		funding_signal.append((funding_data[i] - mvngavg_funding[i]));

	# Simulation to find mean reversal points

	pnl = [0]*len(funding_signal);
	pnl[0] = 1.0;
	p = 0.001
	n = 0.001
	
	while(p< 0.0035):
		while(n< 0.0035):
			for i in range(1,len(funding_signal)):
				if (funding_signal[i-1] > 0):
					if(funding_data[i-1] > p):
						position = -1
					else:
						position = 1
				elif funding_signal[i-1] < 0 :
					if(funding_data[i-1] < -1*n):
						position = 1
					else:
						position = -1
				else:
					position = 0

				if position == 1:
					pnl[i] = (price_data[i] / price_data[i-1]) * pnl[i-1];
				elif position == -1:
					pnl[i] = (price_data[i-1] / price_data[i]) * pnl[i-1];
				else:
					pnl[i] = pnl[i-1]

			if(pnl[-1] > reversal[0]):
				reversal[0] = pnl[-1]
				reversal[1] = p
				reversal[2] = n

			n+=0.0005
		p+=0.0005

	# End of mean reversal simulation

	if(best < reversal[0]):
		best = reversal[0];
		best_window = window;
	   
	window+=1;

# End of best window simulation

print("Best profit:",best)
print("Best window:",best_window)


# Moving average window set using simulations
window = best_window

# Creates EMA of 'funding_data' with 'window'
mvngavg_funding = ema.ewma_vectorized_safe(funding_data, 2/(window+1))

funding_signal = [];

# Signal is positive when funding rate is more than EMA and vice versa
for i in range(len(mvngavg_funding)):
	funding_signal.append((funding_data[i] - mvngavg_funding[i]));

# Profit and Loss calculation

pnl = [0]*len(funding_signal);
pnl[0] = 1.0;

for i in range(1,len(funding_signal)):

	if (funding_signal[i-1] > 0):
		if(funding_data[i-1] > reversal[1]):
			position = -1
		else:
			position = 1
	elif funding_signal[i-1] < 0 :
		if(funding_data[i-1] < -1*reversal[2]):
			position = 1
		else:
			position = -1
	else:
		position = 0

	if position == 1:
		pnl[i] = (price_data[i] / price_data[i-1]) * pnl[i-1];
	elif position == -1:
		pnl[i] = (price_data[i-1] / price_data[i]) * pnl[i-1];
	else:
		pnl[i] = pnl[i-1]



# Creating JSON output

funding_results = {'symbol':price_symbol,'data':[]}
for i in range(len(datenum_price_data)):
	day = {
		'date':md.num2date(datenum_price_data[i]).strftime("%Y-%m-%dT%H:%M:%SZ"),
		'funding_rate':funding_data[i],
		'mvngavg_funding':mvngavg_funding[i],
		'funding_signal':funding_signal[i],
		'wallet':pnl[i],
	}
	funding_results['data'].append(day)

# print(json.dumps(final_data, indent=4))



# Saving the final json

# Save location
path_save_data = "results"
filename_funding_results = "{:s}/funding_results.json".format(path_save_data)

# check if the data path exists
ioh.check_path(path_save_data, create_if_not_exist=True)

# save the data
print("saving data to {:s}".format(filename_funding_results))
with open(filename_funding_results, "w") as f:
	json.dump(funding_results,f)

print("Saved!")




# set up the figure
fig, ax = plt.subplots(4, 1, sharex=True, sharey=False)

# plot stuff
ax[0].grid(linewidth=0.4)
ax[1].grid(linewidth=0.4)
ax[2].grid(linewidth=0.4)
ax[3].grid(linewidth=0.4)

# ax[0].plot(datenum_price_data, price_data, linewidth=0.5)
ax[0].plot(datenum_price_data, price_data, linewidth=0.5)


# def animate(i):
ax[1].plot(datenum_funding_data, funding_data, linewidth=0.5)
ax[1].plot(datenum_price_data, mvngavg_funding, color="r", linewidth=0.5)

ax[2].plot(datenum_price_data, funding_signal, color="r", linewidth=0.5)

ax[3].plot(datenum_price_data[:i], pnl[:i], color="r", linewidth=0.5)

# label axes
ax[0].set_ylabel("Price")
ax[1].set_ylabel("Funding rate")
ax[2].set_ylabel("Funding signal")
ax[3].set_ylabel("Wallet")

# legend
ax[1].legend(["Funding rate","EMA"])

# generate the time axes
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax[0]=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
ax[0].xaxis.set_major_formatter(xfmt)

plt.gcf().set_size_inches(32, 18)
# save the plot
plt.savefig('plots/swap_sentiment.png', bbox_inches='tight')
# show the plot
# ani = animation.FuncAnimation(fig, animate, frames=days, interval=1) 
plt.show()
