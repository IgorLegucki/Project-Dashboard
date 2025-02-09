import scrapy
from scrapy.crawler import CrawlerProcess
import json
import re
import yaml

class JsonWriterPipeline(object):
    def open_spider(self, spider):
        self.file = open('Scraping_result_otomoto.json', 'w', encoding='utf-8-sig')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        item_json = json.dumps(dict(item), indent=4, ensure_ascii=False) + '\n'
        self.file.write(item_json)

        return item

import csv

class CsvWriterPipeline(object):
    def open_spider(self, spider):
        # Otwarcie pliku CSV w trybie zapisu
        self.file = open('Scraping_result_otomoto.csv', 'w', newline='', encoding='utf-8')
        # Tworzenie writera CSV
        self.writer = csv.writer(self.file)
        # Zapis nagłówków (klucze z itemów)
        self.writer.writerow(['brand', 'model', 'price', 'engine', 'KM', 'mileage', 'fuel', 'gearbox', 'year', 'place', 'voivode'])

    def close_spider(self, spider):
        # Zamknięcie pliku po zakończeniu scrapowania
        self.file.close()

    def process_item(self, item, spider):
        # Zapis wartości itemów do pliku CSV
        self.writer.writerow([
            item.get('brand', ''),
            item.get('model', ''),
            item.get('price', ''),
            item.get('engine', ''),
            item.get('KM', ''),
            item.get('mileage', ''),
            item.get('fuel', ''),
            item.get('gearbox', ''),
            item.get('year', ''),
            item.get('place', ''),
            item.get('voivode', '')
        ])
        return item

def load_car_brands(file_path="brands.yaml"):
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)  # Wczytujemy plik YAML
    return data.get("car_brands", [])

def split_car_brand(car_name, car_brands):
    for brand in sorted(car_brands, key=len, reverse=True):  # Sortowanie od najdłuższych
        if car_name.startswith(brand):
            return brand, car_name[len(brand):].strip()
    return "Inne", car_name

class MySpider(scrapy.Spider):
    name = 'mySpider'
    start_urls = [f'https://www.otomoto.pl/osobowe?page={i}' for i in range(1, 7800)]

    custom_settings = {
        'LOG_LEVEL': 'INFO',
        'ITEM_PIPELINES': {'__main__.CsvWriterPipeline': 2},
        #'ITEM_PIPELINES': {'__main__.JsonWriterPipeline': 2},
        'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7',
        'FEEDS': {'quoteresult_otomoto.json': {'format': 'json', 'overwrite': True}}
    }

    def __init__(self, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.car_brands = load_car_brands()

    def parse(self, response):
        listings = response.css('div[data-testid="search-results"]')

        for listing in listings:
            yield self._parse_listing(listing, response)

    def _parse_listing(self, listing, response):

        item = {
            'brand': 'N/A',
            'model': 'N/A',
            'price': 'N/A',
            'engine': '0.0',
            'KM': '0.0',
            'mileage': '0.0',
            'fuel': 'N/A',
            'gearbox': 'N/A',
            'year': 'N/A',
            'place': 'N/A',
            'voivode': 'N/A',
        }

        # Extract the link
        car_name = listing.css('a[target="_self"]::text').get(default='N/A')

        brand, model = split_car_brand(car_name, self.car_brands)

        item['brand'] = brand

        item['model'] = model

        item['price'] = listing.css('h3.ecit9451.ooa-1n2paoq::text').get(default='N/A').replace(' ', '')

        full_text_engine = listing.css('p.ekpvtd0.ooa-1gjazjm::text').get(default='N/A')

        engine_match = re.search(r'(\d[\d\s]*)\s*cm3', full_text_engine)
        power_match = re.search(r'(\d+)\s*KM', full_text_engine)

        if engine_match:
            item['engine'] = engine_match.group(1).strip().replace(' ', '.')

        if power_match:
            item['KM'] = power_match.group(1).replace(' ', '')

        item['mileage'] = listing.css('dd[data-parameter="mileage"]::text').get(default='N/A').replace(' km', '').replace(' ', '.')

        item['fuel'] = listing.css('dd[data-parameter="fuel_type"]::text').get(default='N/A')

        item['gearbox'] = listing.css('dd[data-parameter="gearbox"]::text').get(default='N/A')

        item['year'] = listing.css('dd[data-parameter="year"]::text').get(default='N/A')

        full_text_place = listing.css('p.ooa-oj1jk2::text').get(default='N/A')

        place_match = re.search(r'(.+)\s*\((.+)\)', full_text_place)

        if place_match:
            item['place'] = place_match.group(1).strip()
            item['voivode'] = place_match.group(2).strip()

        return item

process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
    'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7',
})

process.crawl(MySpider)
process.start()