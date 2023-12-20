import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Collection via Web Scraping ---

URL = 'https://krisha.kz/arenda/kvartiry/astana/'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                  ' AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def clean_price(price_text):
    cleaned_price = price_text.replace('\xa0', '').replace('〒', '')
    cleaned_price = ''.join(filter(str.isdigit, cleaned_price))
    return int(cleaned_price) if cleaned_price else None


def extract_details_from_text(text):
    details = {
        'rooms': None,
        'square_meters': None,
        'floor_info': None
    }
    parts = text.split(',')
    if len(parts) >= 3:
        details['rooms'] = parts[0].strip()
        details['square_meters'] = parts[1].strip()
        details['floor_info'] = parts[2].strip()
    return details


data = []


def get_data_from_page(url):
    page = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(page.content, 'html.parser')
    listings = soup.find_all('div', class_='a-card__descr')
    for listing in listings:
        details = {}
        a_tag = listing.find('a', class_='a-card__title')
        if a_tag:
            extracted = extract_details_from_text(a_tag.text)
            details.update(extracted)

        price_div = listing.find('div', class_='a-card__price')
        if price_div:
            details['price'] = clean_price(price_div.text)

        address_div = listing.find('div', class_='a-card__subtitle')
        if address_div:
            details['address'] = address_div.text.strip()

        data.append(details)

def scrape_data():
    for i in range(1, 10):
        page_url = URL + f'?page={i}'
        time.sleep(5)
        get_data_from_page(page_url)


scrape_data()


# --- Data Preprocessing ---

df = pd.DataFrame(data)
df.dropna(inplace=True)  # Drop rows with missing values

# Convert 'rooms' column to represent number of rooms as integer
df['rooms'] = df['rooms'].str.extract('(\d+)').astype(int)

# Extract square meters as float
df['square_meters'] = df['square_meters'].str.extract('(\d+)').astype(float)

# Extract the first number from 'floor_info' to represent the floor
df['floor'] = df['floor_info'].str.extract('(\d+)').astype(float)

# Drop the original 'floor_info' column
df.drop('floor_info', axis=1, inplace=True)

# One-hot encode 'address' column
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[['address']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['address']))
df = pd.concat([df.drop('address', axis=1), encoded_df], axis=1)

# Handle any NaN values (either drop or replace with mean/median)
df.dropna(inplace=True)

# --- Decision Tree Model Building ---

# Split the dataset into training and testing sets
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Initial MSE: {mse}")

# --- Hyperparameter Tuning ---

param_grid = {
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
print(f"MSE after Hyperparameter Tuning: {mse_best}")
# --- Data Visualization ---

# Visualizing the data with actual rental prices

# 1. Distribution of Rental Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True, bins=30)
plt.title('Distribution of Rental Prices')
plt.xlabel('Rental Price (Tenge)')
plt.ylabel('Number of Apartments')
plt.ticklabel_format(style='plain', axis='x')  # Disable the scientific notation on x-axis
plt.show()

# 2. Distribution of Apartment Square Meters
plt.figure(figsize=(10, 6))
sns.histplot(df['square_meters'], kde=True, bins=30)
plt.title('Distribution of Apartment Square Meters')
plt.xlabel('Square Meters')
plt.ylabel('Number of Apartments')
plt.show()

# 3. Relationship between Price and Square Meters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['square_meters'], y=df['price'])
plt.title('Price vs. Square Meters')
plt.xlabel('Square Meters')
plt.ylabel('Rental Price (Tenge)')
plt.ticklabel_format(style='plain', axis='y')  # Disable the scientific notation on y-axis
plt.show()

# 4. Distribution of Apartments by Floors
plt.figure(figsize=(10, 6))
sns.countplot(x=df['floor'])
plt.title('Distribution of Apartments by Floors')
plt.xlabel('Floor')
plt.ylabel('Number of Apartments')
plt.show()

# 5. Average Rental Price by Number of Rooms
plt.figure(figsize=(10, 6))
sns.barplot(x=df['rooms'], y=df['price'])
plt.title('Average Rental Price by Number of Rooms')
plt.xlabel('Number of Rooms')
plt.ylabel('Average Rental Price (Tenge)')
plt.ticklabel_format(style='plain', axis='y')  # Disable the scientific notation on y-axis
plt.show()

# 6. Boxplot for Rental Prices
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['price'])
plt.title('Boxplot for Rental Prices')
plt.ylabel('Rental Price (Tenge)')
plt.ticklabel_format(style='plain', axis='y')  # Disable the scientific notation on y-axis
plt.show()
print("Predicted rental prices:", y_pred_best)
print("Predicted rental prices (initial model):", y_pred)
