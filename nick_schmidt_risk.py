# Nick Schmidt
# SSMIF Risk Coding Challenge
# October 21, 2024

# import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta

class Portfolio:
    def __init__(self, tickers, start_date, end_date, initial_balance=100000):
        
        # initialize the portfolio with the given tickers, start date, end date, and initial balance
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance

        # create a data frame to store the portfolio holdings
        self.holdings = pd.DataFrame(columns=["Unit Cost", "Count", "Total Value", "Max DD", "Volatility", "VaR", "Sharpe"], index = ["CASH"] + tickers)

        # self.holdings.index += "CASH"  (does not function correctly)
        self.holdings["Unit Cost"] = 0.0
        self.holdings["Count"] = 0.0
        self.holdings["Total Value"] = 0.0
        self.holdings["Max DD"] = 0.0
        self.holdings["Volatility"] = 0.0
        self.holdings["VaR"] = 0.0
        self.holdings["Sharpe"] = 0.0

        # add initial cash balance
        self.holdings.loc["CASH", "Count"] = initial_balance
        self.holdings.loc["CASH", "Total Value"] = initial_balance
        self.holdings.loc["CASH", "Unit Cost"] = 1.0

        # create times series metrics_history data frame
        self.metrics_history = pd.DataFrame(columns=["Date","Stock","Max DD", "Volatility", "VaR", "Sharpe","Total Value"])

        # create initial weights of equal value for each stock and cash
        self.weights = {stock: 1/(len(tickers) + 1) for stock in tickers + ["CASH"]}

        # create a dataframe to track the market prices of the S&P 500
        self.market_prices = None

    def buy(self, ticker, count, date):

        """
        This function accepts the ticker, number of shares, and the date the transaction will be made (pass 'day' from the iterating for loop to get the adj close for that specific day)
        The holdings are all updated automatically
        """
        price = self.prices.loc[date][ticker]
        
        #updates the portfolio holdings
        self.holdings.loc[ticker, "Unit Cost"] =  ((self.holdings.loc[ticker, "Unit Cost"] * self.holdings.loc[ticker, "Count"]) + (price * count)) / (self.holdings.loc[ticker, "Count"] + count)
        self.holdings.loc[ticker, "Count"] += count 
        self.holdings.loc[ticker, "Total Value"] = self.holdings.loc[ticker, "Count"] * self.holdings.loc[ticker, "Unit Cost"]
        self.holdings.loc["CASH", "Count"] -= (price * count)
        self.holdings.loc["CASH", "Total Value"] = self.holdings.loc["CASH", "Count"]


    def sell(self, ticker, count, date):
        """
        Similar function to buy(), however, processes a sell order
        """
        price = self.prices.loc[date][ticker]

        #updates portfolio holdings
        self.holdings.loc[ticker, "Count"] -= count 
        self.holdings.loc[ticker, "Total Value"] = self.holdings.loc[ticker, "Count"] * self.holdings.loc[ticker, "Unit Cost"]
        self.holdings.loc["CASH", "Count"] += (price * count)
        self.holdings.loc["CASH", "Total Value"] = self.holdings.loc["CASH", "Count"]
    

    def simulate(self):

        """
        This function process the time series data of adjusted closed prices via for loop to simulate trading days taking place. 
        Data for every stock is initially downloaded using yf.download(), and cleaned into a data frame self.prices for easier processing
        Examples to access specific data are provided
        """

        # allow for 30 days before to calculate metrics
        start_date = pd.to_datetime(self.start_date) - timedelta(days=30)
        start_date = start_date.strftime("%Y-%m-%d")

        # import data for each stock and store in self.prices
        data = yf.download(tickers=self.tickers, start = start_date, end = self.end_date) 
        self.prices = pd.DataFrame(columns=self.tickers) 

        # download S&P 500 data to use as a benchmark
        sp500 = yf.download("^GSPC", start=start_date, end=self.end_date)["Adj Close"]
        self.market_prices = sp500

        # data filtering 
        for ticker in self.tickers:
            self.prices[ticker] = data["Adj Close"][ticker]

        # buy initial holdings (equal value for each stock and cash)
        for stock in self.tickers:
            price_share = self.prices[stock][self.prices.index[0]]
            self.buy(stock, self.initial_balance/(len(self.tickers)+1) // price_share, self.prices.index[0])


        # for loop simulation 
        for day in self.prices.index[self.prices.index >= self.start_date]: # iterate over each day in the data frame

            self.rebalance_portfolio(day) #re-balance portfolio daily
            self.calculate_metrics(day) #calculate metrics daily

        # set date as index for metrics_history   
        self.metrics_history.set_index("Date", inplace=True)


    def calculate_metrics(self,day):
        """
        1. Calculate the metrics for each individual stock in the portfolio
        2. Post these updated metrics to the portfolio data frame
        3. Append the portfolio risk metrics to a time series data frame documenting the historical values along with portfolio value information for each day
        """

        # caculate metrics from the last 30 days
        start_date = day - timedelta(days=30)
        end_date = day
        
        # prepare all metrics that will be calculated for portfolio and stocks
        metrics = ['Max DD', 'Volatility', 'VaR', 'Sharpe']
        stock_metrics = {metric:[] for metric in metrics}
        stock_weights = [self.weights[stock] for stock in self.tickers]

        # calculate risk metrics for each stock over 30 day period
        for stock in self.tickers + ["Market"]:
            
            if stock != "Market":
                # get stock data over the last 30 days
                stock_data = self.prices[stock][start_date:end_date]

            else:
                # get S&P 500 data over the last 30 days
                stock_data = self.market_prices[start_date:end_date]

            # calculate 95% Value at Risk using historical data 
            returns = stock_data.pct_change().dropna()
            confidence_level = 95
            VaR = -np.percentile(returns.fillna(0), 100 - confidence_level)
            

            # calculate annualized volatility (standard deviation of returns)
            volatility = returns.std() * np.sqrt(252) # annualized volatility
            

            # calculate Sharpe Ratio (risk adjusted return)
            expected_return = returns.mean() * 252 # annualized expected return 
            risk_free_rate = 0.0343 # 10 year treasury bond rate in mid 2022
            sharpe_ratio = (expected_return - risk_free_rate) / volatility
            

            # calculate Max Drawdown (maximum distance from peak price to lowest price)
            peak_value = stock_data.cummax() # from CHATGPT
            all_differences = (stock_data - peak_value) / peak_value # each values difference from the max value
            max_dd = all_differences.min() * 100
            

            if stock != "Market":

                # update holdings with currect risk metrcs
                self.holdings.loc[stock, "VaR"] = VaR
                self.holdings.loc[stock, "Volatility"] = volatility
                self.holdings.loc[stock, "Sharpe"] = sharpe_ratio
                self.holdings.loc[stock, "Max DD"] = max_dd

                # append to metrics_history data frame
                new_metrics = [day, stock, max_dd, volatility, VaR, sharpe_ratio, self.holdings.loc[stock, "Total Value"]]
                self.metrics_history.loc[len(self.metrics_history)] = new_metrics
                
                # add each metric to stock_metrics, which will be used to calculate the portfolio metrics
                stock_metrics['VaR'].append(VaR)
                stock_metrics['Volatility'].append(volatility)
                stock_metrics['Sharpe'].append(sharpe_ratio)
                stock_metrics['Max DD'].append(max_dd)

            else:
                # add risk metrics to market_metrics data DataFrame
                new_metrics = [max_dd, volatility, VaR, sharpe_ratio]
                current_price = self.market_prices[day]
                self.metrics_history.loc[len(self.metrics_history)] = [day, "Market"] + new_metrics + [current_price]


        # find the metrics for the portfolio by mutliplying the stock metrics by the weights and adding them together
        total_portfolio_metrics = []
        for metric in metrics:
            total_portfolio_metrics.append(np.dot(stock_metrics[metric], stock_weights))
        
        # append the portfolio metrics to the metrics_history data frame, along with the total value of the portfolio
        self.metrics_history.loc[len(self.metrics_history)] = [day, "Portfolio"] + total_portfolio_metrics + [sum(self.holdings["Total Value"])]


    def rebalance_portfolio(self,day,risk_metric='Sharpe', target_value=1.5,):
        """
        Rebalance the portfolio to achieve the desired risk tolerance. Please state which metrics you plan to use. 
        """
        import numpy as np
        from datetime import timedelta

        # My plan is to use Monte Carlo Simulations and randomly generate a set amounts of weights, and then
        # calculate what the Sharpe Ratio would have been over the past 30 days. I will use the Sharpe Ratios
        # from the self.holdings data frame to find each Sharpe Ration for each set of weights. Then, I will
        # find which weights generate a Sharpe Ratio within the bounds of the target value plus and minus 0.25. 
        # Then, I will find the max Sharpe Ratio from that group and readjust the portfolio based on those weights.

        # if the day is not the first day, update the unit cost and total value of each stock, based on the 
        # historical growth of the stock over the past day
        row = self.prices.index.get_loc(day)
        if row > 0:
            for stock in self.tickers:
                # find percent change in stock price
                new_price = self.prices.iloc[row][stock] 
                old_price = self.prices.iloc[row-1][stock]
                change = (new_price - old_price) / old_price

                # adjust holdings based on the growth of the stock
                self.holdings.loc[stock, "Unit Cost"] *= (1 + change)
                self.holdings.loc[stock, "Total Value"] = self.holdings.loc[stock, "Count"] * self.holdings.loc[stock, "Unit Cost"]


        # create a list for each stock's weight and Sharpe Ratio
        stock_weights = [self.weights[stock] for stock in ["CASH"] + self.tickers]
        sharpe_ratios = [self.holdings.loc[stock, risk_metric] for stock in ["CASH"] + self.tickers]
        new_weights = []

        # until there are 500 sets of weigths, generate random weights and find the Sharpe Ratio
        while len(new_weights) < 500:

            # generate a random number for each stock from 1-300 and divide each by total to get a weight
            random_numbers = [np.random.randint(1,300) for stock in ["CASH"] + self.tickers]
            possible_weights = [number / sum(random_numbers) for number in random_numbers]

            # find the new Sharpe Ratio and add to the list of new weights and sharpe ratios
            new_sharpe_ratio = np.dot(possible_weights, sharpe_ratios)
            new_weights += [[possible_weights,new_sharpe_ratio]]

        # find weights that would lead to a Sharpe Ratio within the target value +/- 0.25
        new_weights = [weight for weight in new_weights if weight[1] > target_value-0.25 and weight[1] < target_value + 0.25]
        
        # make sure there are weights that meet the criteria, if not do no rebalance
        try:
            # used ChatGPT to use the max function with a multi-dimensional list
            best_sharpe_ratio_and_weights = max(new_weights, key=lambda x: x[1])

            # only rebalance is in the target ratio range
            if best_sharpe_ratio_and_weights[1] > target_value-0.25 and best_sharpe_ratio_and_weights[1] < target_value + 0.25:
                
                best_weights = best_sharpe_ratio_and_weights[0]
            
                # find current values of each stock and total value of all the stocks
                current_values = {stock: self.holdings.loc[stock, "Total Value"] for stock in ["CASH"] + self.tickers}
                total_value = sum(current_values.values())
                new_values = {stock:0 for stock in ["CASH"] + self.tickers}

                # find the new values of each stock based on the best weights and total value
                for num,stock in enumerate(["CASH"] + self.tickers):
                    new_values[stock] = total_value * best_weights[num]
                
                # find the difference between the current value and the new value
                changes = {stock:round(new_values[stock] - current_values[stock],2) for stock in ["CASH"] + self.tickers}

                # find the amount of shares to buy or sell based on the changes
                amount_shares = {stock: changes[stock] / self.prices.loc[day][stock] for stock in self.tickers}

                # buy or sell stocks based on the changes
                for stock in self.tickers:
                    if amount_shares[stock] > 0:
                        self.buy(stock, amount_shares[stock], day)
                    else:
                        self.sell(stock, -amount_shares[stock], day)
        except:
            pass

    def display_data(self):
        """
        Display graphs that show portfolio value, returns, and the risk metrics.  Additional display the data frame with the latest portfolio holdings.
        """

        import numpy as np
        import matplotlib.pyplot as plt

        # graph the portfolio value over the given period of time
        portfolio_value = self.metrics_history[self.metrics_history["Stock"] == "Portfolio"]["Total Value"]
        plt.figure(figsize=(10,5))
        plt.plot(portfolio_value, label="Portfolio")

        # dowload and plot S&P 500 data to use as a benchmark and plot it
        sp500 = yf.download("^GSPC", start=self.start_date, end=self.end_date)["Adj Close"]
        sp500 = sp500 / sp500.iloc[0] * self.initial_balance
        plt.plot(sp500, label="S&P 500")

        # label the graph
        plt.title("Portfolio vs. S&P 500 Growth Over Time")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
        
        # graph the portfolio risk metrics and benchmark risk metrics over the given time period
        portfolio_metrics = self.metrics_history[self.metrics_history["Stock"] == "Portfolio"]
        market_metrics = self.metrics_history[self.metrics_history["Stock"] == "Market"] 

        # cycle through all metrics and graph each one
        for metric in ['Max DD', 'Volatility', 'VaR', 'Sharpe']:

            # plot the metric for the portfolio and the benchmark
            plt.figure(figsize=(10,5))
            plt.plot(portfolio_metrics[metric], label="Portfolio")
            plt.plot(market_metrics[metric], label="Market")

            # label the graph
            plt.title(f"{metric} Over Time")
            plt.xlabel("Date")
            plt.ylabel(metric)
            plt.legend()
            plt.show()

        # The plot of the Sharpe Ratio of the portfolio vs. the Sharpe Ratio of the benchmark over the specified
        # time period illustrates how reblancing of the portfolio restricted the variation of the Sharpe Ratio 
        # compared to the Sharpe Ratio of the S&P 500. The Sharpe Ratio of the portfolio is more stable.

        # printout of the last holding snapshot
        print("\n\nLatest Portfolio Holdings")
        print(self.holdings,"\n\n")


port = Portfolio(tickers=["TSLA","IBM","DIS","MA"], start_date="2020-01-01", end_date="2023-12-31", initial_balance=100000) # "2023-12-31"

port.simulate()
port.display_data()