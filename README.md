# Auto-Balancing-Portfolio
Coding Challenge for Stevens Student Managed Investment Fund

In this challenge, I was tasked to create a class that tracks specific metrics of a portfolio of stocks over a given timeframe. The risk metrics I used to to analyze the portfolio were volatility, 95% Value at Risk, Sharpe Ratio, and Maximum Drawdown.

The second part of this project was creating a system to automatically adjust the porfolio to minimize risk, while still generating profits. I created a function to randomly generate weights for each of the stocks and find the new Sharpe Ratio. I found the weights that had this ratio in a range that satisfied the users needs, and made the necessary adjustments to the portfolio.

I also created a function called display_data to visualize the risk metrics and growth of a portfolio versus the market.