# Google Maps API

## Quick Start

安裝 `googlemaps` python package 即可

```bash
pip install googlemaps
```

## Usage

有兩種模式：`place` 跟 `search`

- `place`: 搜尋指定地點
- `search`: 搜尋一個比較大的名詞 e.g. 餐廳、咖啡廳等

```bash
python gmap_scraper.py place --name "臺北市立美術館"
python gmap_scraper.py place --name "臺北玫瑰園"
python gmap_scraper.py search --keyword "餐酒"
python gmap_scraper.py search --keyword "餐廳"
python gmap_scraper.py search --keyword "咖啡廳"
```