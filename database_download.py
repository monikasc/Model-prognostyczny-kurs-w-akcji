import yfinance as yf
stock = yf.Ticker("MSFT")
print(stock.info)
history = stock.history(period='max', interval='1d', )
history.to_csv('C:\\Users\\tqgr38\\Desktop\\praca lic\\datasets\\MSFT.csv')