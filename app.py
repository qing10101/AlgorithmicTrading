import pandas as pd
import matplotlib.pyplot as plt


# Function to calculate Average True Range (ATR) for volatility adjustments
def calculate_atr(df, n=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-Cp'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=n).mean()
    return df


# Modified ADX function with adaptive window based on volatility
def calculate_adx(df, n=14):
    df = calculate_atr(df, n)  # Use ATR for volatility measure
    df['DM+'] = (df['High'] - df['High'].shift(1)).clip(lower=0)
    df['DM-'] = (df['Low'].shift(1) - df['Low']).clip(lower=0)
    df['TRn'] = df['TR'].rolling(window=n).sum()
    df['DMn+'] = df['DM+'].rolling(window=n).sum()
    df['DMn-'] = df['DM-'].rolling(window=n).sum()
    df['DI+'] = 100 * (df['DMn+'] / df['TRn'])
    df['DI-'] = 100 * (df['DMn-'] / df['TRn'])
    df['DIdiff'] = (df['DI+'] - df['DI-']).abs()
    df['DIsum'] = df['DI+'] + df['DI-']
    df['DX'] = 100 * (df['DIdiff'] / df['DIsum'])
    df['ADX'] = df['DX'].rolling(window=n).mean()
    return df


# MACD with longer windows for trend-following
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['EMA_short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    return df


# RSI with dynamic window adjustment based on volatility
def calculate_rsi(df, window=14):
    df = calculate_atr(df, window)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    return df


# Function to calculate long-term trend using moving average crossover
def calculate_moving_average(df, short_window=50, long_window=200):
    df['MA_short'] = df['Close'].rolling(window=short_window).mean()
    df['MA_long'] = df['Close'].rolling(window=long_window).mean()
    return df


# Generate buy/sell signals based on enhanced criteria
def generate_signals(df):
    df['Buy_Signal'] = ((df['MACD'] > df['Signal']) & (df['RSI'] < 30) &
                        (df['ADX'] > 20) & (df['MA_short'] > df['MA_long']))

    df['Sell_Signal'] = ((df['MACD'] < df['Signal']) & (df['RSI'] > 70) &
                         (df['ADX'] > 20) & (df['MA_short'] < df['MA_long']))

    return df


# Function to predict trend based on the latest data
def predict_trend(df):
    # Get the first row (most recent data)
    latest_data = df.iloc[0]

    trend = "Neutral/Sideways"

    # Conditions to predict an upward trend
    if latest_data['MACD'] > latest_data['Signal'] and latest_data['RSI'] < 70 and latest_data['MA_short'] > \
            latest_data['MA_long']:
        if latest_data['ADX'] > 20:  # Strong trend
            trend = "Strong Upward Trend"
        else:  # Weak trend
            trend = "Weak Upward Trend"

    # Conditions to predict a downward trend
    elif latest_data['MACD'] < latest_data['Signal'] and latest_data['RSI'] > 30 and latest_data['MA_short'] < \
            latest_data['MA_long']:
        if latest_data['ADX'] > 20:  # Strong trend
            trend = "Strong Downward Trend"
        else:  # Weak trend
            trend = "Weak Downward Trend"

    # Print the analysis and trend prediction
    print(f"Date: {latest_data['Date']}")
    print(f"Close Price: {latest_data['Close']}")
    print(f"MACD: {latest_data['MACD']}, Signal: {latest_data['Signal']}")
    print(f"RSI: {latest_data['RSI']}")
    print(f"ADX: {latest_data['ADX']}")
    print(f"Short MA: {latest_data['MA_short']}, Long MA: {latest_data['MA_long']}")
    print(f"Trend Prediction: {trend}")

    return trend


# Function to plot stock data, indicators, and buy/sell signals
def plot_chart(df):
    buy_signals = df[df['Buy_Signal']]
    sell_signals = df[df['Sell_Signal']]

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

    # Plot closing price and buy/sell signals
    ax1.plot(df['Date'], df['Close'], label='Close Price')
    ax1.plot(buy_signals['Date'], buy_signals['Close'], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    ax1.plot(sell_signals['Date'], sell_signals['Close'], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    ax1.set_title('Stock Price and Trading Signals')
    ax1.legend()

    # Plot MACD and signal line
    ax2.plot(df['Date'], df['MACD'], label='MACD')
    ax2.plot(df['Date'], df['Signal'], label='Signal')
    ax2.set_title('MACD')
    ax2.legend()

    # Plot RSI
    ax3.plot(df['Date'], df['RSI'], label='RSI')
    ax3.axhline(70, color='r', linestyle='--', linewidth=0.5)
    ax3.axhline(30, color='g', linestyle='--', linewidth=0.5)
    ax3.set_title('RSI')
    ax3.legend()

    # Plot ADX
    ax4.plot(df['Date'], df['ADX'], label='ADX')
    ax4.set_title('ADX')
    ax4.legend()

    # Plot moving averages for trend identification
    ax5.plot(df['Date'], df['MA_short'], label='50-day MA', color='b')
    ax5.plot(df['Date'], df['MA_long'], label='200-day MA', color='orange')
    ax5.set_title('Moving Averages (Trend Confirmation)')
    ax5.legend()

    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Main execution block
if __name__ == "__main__":
    # Read stock data from a CSV file
    df = pd.read_csv('tsm.csv')

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Reverse the DataFrame so the most recent date is at the bottom (for consistency in rolling calculations)
    df = df.iloc[::-1].reset_index(drop=True)

    # Calculate technical indicators: ADX, MACD, RSI, moving averages
    df = calculate_adx(df)
    df = calculate_macd(df)
    df = calculate_rsi(df)
    df = calculate_moving_average(df)

    # Reverse the DataFrame back to its original order
    df = df.iloc[::-1].reset_index(drop=True)

    # Generate buy/sell signals based on enhanced criteria
    df = generate_signals(df)

    # Plot the stock chart with trading signals
    plot_chart(df)

    # Predict trend based on the latest data (top row)
    trend_prediction = predict_trend(df)
