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

# check if price and funding cryptocurrencies match

if(article_symbol != funding_symbol):
    print("Article Symbol:",article_symbol)
    print("Funding Symbol:",funding_symbol)
    sys.exit("Funding and Article symbols don't match! Please match the symbols and re-run loading scripts.")




# Loading the article signal

article_signal = []
for i in range(len(article_results['data'])):
    article_signal.append(article_results['data'][i]['ratio_signal'])



# Loading the funding signal

funding_signal = []
for i in range(len(funding_results['data'])):
    funding_signal.append(funding_results['data'][i]['funding_signal'])

temp = funding_signal[-1]
funding_signal = funding_signal[::3]

# Last date cut off during slicing is attached again
funding_signal.append(temp)


# Scaling funding_signal to be comparable in value to article_signal

funding_signal = [(funding_signal[i] - np.mean(funding_signal))/np.std(funding_signal) for i in range(len(funding_signal))]




# Simulation to find the best scaling factor - 'k'

# Variable that holds the best scenario - [Profit,Factor,Positive Bias,Negative Bias]
best_factor = [-1000,0,0,0]

for k in np.arange(0,10,0.05):

    # S3 = S1 + K * S2 (where S1, S2, and S3 are Funding, Article, and Combined Signals respectively)

    combined_signal = [k*article_signal[0]]
    combined_signal.extend([funding_signal[i-1] + k*article_signal[i] for i in range(1,len(funding_signal))])

    best = [-1000,0,0]

    # Simulation to find the bias in articles

    p = 0;

    while(p<=2):
        n=0;
        while(n<=2):
            pnl = [0]*len(combined_signal);
            pnl[0] = 1.0;
            for i in range(1,len(combined_signal)):

                # Long position
                if (combined_signal[i-1] > p):
                    position = 1

                # Short position
                elif combined_signal[i-1] < -1*n :
                    position = -1

                # No position
                else:
                    position = 0

                if position == 1:
                    pnl[i] = (price_data[i] / price_data[i-1]) * pnl[i-1];
                elif position == -1:
                    pnl[i] = (price_data[i-1] / price_data[i]) * pnl[i-1];
                else:
                    pnl[i] = pnl[i-1]

            if(best[0]<pnl[-1]):
                best[0]=pnl[-1];
                best[1]=p;
                best[2]=n;

            n+=0.1;
        p+=0.1;

    # End of bias simulation

    if(best[0] > best_factor[0]):
        best_factor[0] = round(best[0],3);
        best_factor[1] = round(k,3);
        best_factor[2] = round(best[1],2);
        best_factor[3] = round(best[2],2);
        print("Best profit for (",best_factor[1],") as scaling factor is:",best_factor[0])



print("Best profit:",best_factor[0]);
print("Best factor:",best_factor[1]);
print("Positive bias:",best_factor[2]);
print("Negative bias: ",best_factor[3]);



# Profit and Loss calculation

combined_signal = [best_factor[1]*article_signal[0]]

combined_signal.extend([funding_signal[i-1] + best_factor[1]*article_signal[i] for i in range(1,len(funding_signal))])

pnl = [0]*len(combined_signal);
pnl[0] = 1.0;

for i in range(1,len(combined_signal)):

    # Long position
    if (combined_signal[i-1] > best_factor[2]):
        position = 1

    # Short position
    elif combined_signal[i-1] < -1*best_factor[3] :
        position = -1

    # No position
    else:
        position = 0

    if position == 1:
        pnl[i] = (price_data[i] / price_data[i-1]) * pnl[i-1];
    elif position == -1:
        pnl[i] = (price_data[i-1] / price_data[i]) * pnl[i-1];
    else:
        pnl[i] = pnl[i-1]


# Creating JSON output

combined_results = {'symbol':price_symbol,'scaling_factor':best_factor[1],'data':[]}

for i in range(len(datenum_price_data)):
    day = {
        'date':md.num2date(datenum_price_data[i]).strftime("%Y-%m-%dT%H:%M:%SZ"),
        'combined_signal':combined_signal[i],
        'wallet':pnl[i],
    }
    combined_results['data'].append(day)

# print(json.dumps(combined_results['data'][0], indent=4))




# Saving the final json

# Save location
path_save_data = "results"
filename_combined_results = "{:s}/combined_results.json".format(path_save_data)

# check if the data path exists
ioh.check_path(path_save_data, create_if_not_exist=True)

# save the data
print("saving data to {:s}".format(filename_combined_results))
with open(filename_combined_results, "w") as f:
	json.dump(combined_results,f)

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

ax[1].plot(datenum_price_data, article_signal, color="b", linewidth=0.5)

ax[1].plot(datenum_price_data, funding_signal, color="r", linewidth=0.5)

ax[2].plot(datenum_price_data, combined_signal, color="r", linewidth=0.5)

ax[3].plot(datenum_price_data, pnl, color="r", linewidth=0.5)


# label axes
ax[0].set_ylabel("Price")
ax[1].set_ylabel("Signals")
ax[2].set_ylabel("Combined Signal")
ax[3].set_ylabel("Wallet")

# legend
ax[1].legend(["Article","Funding (scaled)"])

# generate the time axes
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
ax[0]=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d %H:%M')
ax[0].xaxis.set_major_formatter(xfmt)

plt.gcf().set_size_inches(32, 18)

# save the plot
plt.savefig('plots/combined_sentiment.png', bbox_inches='tight')

# show the plot
# ani = animation.FuncAnimation(fig, animate, frames=days, interval=1) 
plt.show()


