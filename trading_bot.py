import pandas as pd
import numpy as np
import openpyxl as pyxl

# training data is from yahoo finance VOO data 2019-2023
trn_data = pd.read_excel("/Users/ianrichthammer/Library/Containers/com.microsoft.Excel/Data/Downloads/VOO_New.xlsx")

print(trn_data.shape)


# converting csv into classes

class data_point:
    def __init__(self, date, open_price, high, low, close_price, volume):
        self.date = date
        self.open_price = open_price
        self.high = high
        self.low = low
        self.close_price = close_price
        self.volume = volume


# List to store instances of data_point (source: openai)
data_points = []

# Iterate over each row in trn_data and create data_point objects (source: openai)
for index, row in trn_data.iterrows():
    dp = data_point(row['Date'], row['Open'], row['High'], row['Low'], row['Close'], row['Volume'])
    data_points.append(dp)


class Bot:
    def __init__(self, money=10000, shares=0):
        self.money = money
        self.shares = shares

    def account_value(self, point):
        price = data_points[point].close_price
        return self.money + (self.shares * price)

    # excs is excess

    def buy_open(self, point, quant):
        price = data_points[point].open_price
        trans_val = price * quant
        if trans_val > self.money:
            quant = self.money // price
            trans_val = price * quant
        self.money -= trans_val
        self.shares += quant
        print(f"On {data_points[point].date}, {quant} shares were bought at a price of ${price}")

    def buy_close(self, point, quant):
        price = data_points[point].close_price
        trans_val = price * quant
        if trans_val > self.money:
            quant = self.money // price
            trans_val = price * quant
        self.money -= trans_val
        self.shares += quant
        print(f"On {data_points[point].date}, {quant} shares were bought at a price of ${price}")

    def sell_open(self, point, quant):
        price = data_points[point].open_price
        if quant > self.shares:
            quant = self.shares
        value = price * quant
        self.money += value
        self.shares -= quant
        print(f"On {data_points[point].date}, {quant} shares were sold at a price of ${price}")

    def sell_close(self, point, quant):
        price = data_points[point].close_price
        if quant > self.shares:
            quant = self.shares
        value = price * quant
        self.money += value
        self.shares -= quant
        print(f"On {data_points[point].date}, {quant} shares were sold at a price of ${price}")


bot = Bot()

for i in range(len(data_points)):
    z = np.random.randint(0, 2)
    if z == 0:
        bot.buy_open(i, np.random.randint(0, 30))
    else:
        bot.sell_open(i, np.random.randint(0, 30))

print(bot.account_value(1000))
