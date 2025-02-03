from alpaca_trade_api import REST
import pandas as pd
import requests


def predict_and_trade(symbol, model, time_steps):
    api = REST('AKSAMRI5EBZ4VBBDC69D', 'yOjOKPjZLKkACovRvadEeVcFztuibcThWeXwGoJEC', base_url='https://paper-api.alpaca.markets')

    try:
        # Fetch the latest historical data using get_bars with valid timeframe "1Min"
        data = api.get_bars(symbol, timeframe="1Min", limit=time_steps).df
        if data.empty:
            print(f"No data returned for {symbol}.")
            return
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return

    # Ensure data is properly sorted by timestamp
    data = data.sort_index()

    # Extract closing prices
    closing_prices = data['close'].values.reshape(-1, 1)
    
    # Predict the next price using the model
    predicted_price = model.predict(closing_prices[-time_steps:].reshape(1, time_steps, 1))

    if predicted_price is not None and len(predicted_price) > 0:
        predicted_price = predicted_price[0][0]
        current_price = closing_prices[-1][0]

        if predicted_price > current_price * 1.01:
            print(f"Buying {symbol} at price {current_price}")
        elif predicted_price < current_price * 0.99:
            print(f"Selling {symbol} at price {current_price}")
        else:
            print(f"Hold position: Predicted price is close to current price.")
    else:
        print(f"Prediction failed for {symbol}.")



