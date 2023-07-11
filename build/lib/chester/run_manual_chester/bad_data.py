import json
import re

import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class FlightDataParser:
    def __init__(self):
        self.df = pd.DataFrame()

    def from_json_to_pandas(self, json_file):
        with open(json_file, 'r') as f:
            flight_data = json.load(f)

        self.df = pd.DataFrame(flight_data)
        self.df['created_at'] = pd.to_datetime(self.df['created_at'])
        self.df.sort_values(by='created_at', ascending=False, inplace=True)
        # self.df.drop_duplicates(subset=['from_date', 'to_date', 'destination'], keep='first', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df['from_date'] = pd.to_datetime(self.df['from_date'], format='%b %d', errors='coerce')
        self.df['from_date'] = self.df['from_date'].apply(lambda x: x.replace(year=2023) if pd.notnull(x) else x)
        self.df['to_date'] = pd.to_datetime(self.df['to_date'], format='%b %d', errors='coerce')
        self.df['to_date'] = self.df['to_date'].apply(lambda x: x.replace(year=2023) if pd.notnull(x) else x)

    def parse_flights(self, text):
        flight_pattern = re.compile(
            r'(\d{1,2}:\d{2}\s*[AP]M)(\s*\+\d)?\s*–\s*'
            r'(\d{1,2}:\d{2}\s*[AP]M)(\s*\+\d)?\n'
            r'(?:Separate tickets booked together)?\s*([^,\n]+)(?:,)?[^,\n]*\n'
            r'(\d\s*hr\s*\d{1,2}\s*min)'
        )

        flights = []
        for match in flight_pattern.finditer(text):
            departure_time = match.group(1)
            if match.group(2):
                departure_time += match.group(2)
            arrival_time = match.group(3)
            if match.group(4):
                arrival_time += match.group(4)
            airline = match.group(5).strip()
            duration = match.group(6)
            price_pattern = re.compile(r'₪(\d+,*\d*)')
            price_match = price_pattern.search(text, match.end())
            if price_match:
                price = int(price_match.group(1).replace(',', ''))
                flights.append({
                    'departure_time': departure_time,
                    'arrival_time': arrival_time,
                    'airline': airline,
                    'duration': duration,
                    'origin': 'TLV',
                    'destination': 'ATH',
                    'price': price
                })

        return pd.DataFrame(flights)

    def explode_flights(self):
        parsed_flights = []
        for index, row in self.df.iterrows():
            flights_df = self.parse_flights(row['content'])
            flights_list = flights_df.to_dict('records')

            for flight in flights_list:
                flight['from_date'] = row['from_date']
                flight['to_date'] = row['to_date']
                flight['destination'] = row['destination']
                flight['created_at'] = row['created_at']

            parsed_flights.extend(flights_list)

        columns = ['from_date', 'to_date', 'destination', 'created_at', 'departure_time', 'arrival_time', 'airline',
                   'duration', 'origin', 'price']
        self.df = pd.DataFrame(parsed_flights, columns=columns)


def load_flight_data():
    json_file_path = "/Users/amitosi/PycharmProjects/databots/databots/researcher/google_flights/flights.json"
    parser = FlightDataParser()
    parser.from_json_to_pandas(json_file_path)
    parser.explode_flights()
    return parser.df


df = load_flight_data()

# Performing EDA for Various Features
from chester.run import full_run as fr
from chester.run import user_classes as uc

fr.run(uc.Data(df=df))
