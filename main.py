from data_fetcher import fetch_data, prepare_data
from model_builder import build_model
from trading_strategy import predict_and_trade

def main():
    # Step 1: Fetch historical data
    stock_data = fetch_data("AAPL", "2020-01-01", "2023-01-01")
    
    # Step 2: Prepare data for the model
    processed_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    time_steps = 60
    X, y = prepare_data(processed_data, time_steps)

    # Step 3: Build and train the model
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=20, batch_size=32)

    # Step 4: Start live trading
    predict_and_trade("AAPL", model, time_steps)

if __name__ == "__main__":
    main()
