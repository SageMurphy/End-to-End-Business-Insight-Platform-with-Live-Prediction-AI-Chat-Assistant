import pandas as pd
import numpy as np
import time
from datetime import datetime

def generate_realtime_data(num_rows=1):
    """Generates simulated sales data."""
    products = ['Electronics', 'Clothing', 'Books', 'Home Goods']
    payment_methods = ['Credit Card', 'Debit Card', 'UPI', 'Net Banking']

    data = []
    for _ in range(num_rows):
        timestamp = datetime.now()
        product_type = np.random.choice(products)
        price = round(np.random.uniform(10, 500), 2)
        num_clicks = np.random.randint(1, 100)
        payment_method = np.random.choice(payment_methods)
        customer_id = np.random.randint(1000, 9999)
        data.append([timestamp, product_type, price, num_clicks, payment_method, customer_id])

    df = pd.DataFrame(data, columns=['timestamp', 'product_type', 'price', 'num_clicks', 'payment_method', 'customer_id'])
    return df

if __name__ == "__main__":
    while True:
        new_data = generate_realtime_data()
        print(new_data)
        # In a real application, you would likely append this to a larger DataFrame
        # or stream it to your dashboard.
        time.sleep(1) # Simulate data generation every 1 second