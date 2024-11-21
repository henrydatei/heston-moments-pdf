import pandas as pd

def align_to_business_days(df):
    # Check if the DataFrame has a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    
    # Ensure the DataFrame is sorted by date
    df = df.sort_index()

    # Align the first point
    while True:
        # Get the first date in the index
        first_date = df.index[0]
        
        # Get the first business day of that month
        first_business_day = pd.date_range(first_date, periods=1, freq='BMS')[0]
        
        if first_date == first_business_day:
            break
        else:
            # Drop the first row if the first date is not the first business day
            df = df.iloc[1:]

    # Align the last point
    while True:
        # Get the last date in the index
        last_date = df.index[-1]
        
        # Get the last business day of that month
        last_business_day = pd.date_range(last_date, periods=1, freq='BM')[0]
        
        if last_date == last_business_day:
            break
        else:
            # Drop the last row if the last date is not the last business day
            df = df.iloc[:-1]
    
    return df