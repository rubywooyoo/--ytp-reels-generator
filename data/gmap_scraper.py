import os
import csv
import re
import argparse
import googlemaps

# 填入你的 API Key
GMAP_API_KEY = 'AIzaSyAcGRvBWQsl9MiE84CVta4vH46QPsnbS6o'
gmaps = googlemaps.Client(key=GMAP_API_KEY)

CSV_HEADERS = ['店名', '地址', '平均評分', '總評分人數', '步行距離', '預計步行時間', '評論1', '評論2', '評論3', '圖片資料夾']


def fetch_details_and_save(place_id, name, dest, origin, output_root, folder_prefix, writer):
    """Fetch details, download photos, and write one row to the CSV writer."""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", name)
    folder_name = f"{folder_prefix}_{safe_name}"
    place_folder = os.path.join(output_root, folder_name)
    os.makedirs(place_folder, exist_ok=True)

    # Walking distance
    if origin:
        matrix = gmaps.distance_matrix(
            origins=[origin],
            destinations=[(dest['lat'], dest['lng'])],
            mode="walking"
        )
        element = matrix['rows'][0]['elements'][0]
        dist_text = element['distance']['text']
        duration_text = element['duration']['text']
    else:
        dist_text = duration_text = "N/A"

    # Details: reviews & photos
    details = gmaps.place(
        place_id=place_id,
        fields=['review', 'rating', 'photo', 'formatted_address'],
        language='zh-TW'
    )['result']

    # Reviews
    reviews = details.get('reviews', [])
    review_list = [f"[{r.get('rating')}⭐] {r.get('text','').replace(chr(10),' ')}" for r in reviews[:3]]
    while len(review_list) < 3:
        review_list.append("暫無評論")

    # Photos
    photo_count = 10
    for idx, photo in enumerate(details.get('photos', [])[:photo_count]):
        save_path = os.path.join(place_folder, f"photo_{idx+1}.jpg")
        with open(save_path, "wb") as f:
            for chunk in gmaps.places_photo(photo['photo_reference'], max_width=800):
                if chunk:
                    f.write(chunk)

    if writer:
        writer.writerow({
            '店名': name,
            '地址': details.get('formatted_address', ''),
            '平均評分': details.get('rating', 0),
            '總評分人數': '',
            '步行距離': dist_text,
            '預計步行時間': duration_text,
            '評論1': review_list[0],
            '評論2': review_list[1],
            '評論3': review_list[2],
            '圖片資料夾': folder_name
        })

    return folder_name


def mode_search(args):
    """Search nearby places by keyword and save results."""
    origin = (args.lat, args.lng)
    output_root = args.output or re.sub(r'[\\/*?:"<>| ]', "_", args.keyword) + "_results"
    os.makedirs(output_root, exist_ok=True)

    csv_path = os.path.join(output_root, "data.csv")
    places_result = gmaps.places_nearby(
        location=origin,
        radius=args.radius,
        keyword=args.keyword,
        language='zh-TW'
    )
    results = places_result.get('results', [])[:args.limit]
    print(f"找到 {len(results)} 筆結果，關鍵字：{args.keyword}")

    with open(csv_path, mode='w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for i, place in enumerate(results, start=1):
            name = place.get('name', '')
            print(f"  [{i}/{len(results)}] {name}")
            try:
                fetch_details_and_save(
                    place_id=place['place_id'],
                    name=name,
                    dest=place['geometry']['location'],
                    origin=origin,
                    output_root=output_root,
                    folder_prefix=str(i),
                    writer=writer
                )
            except Exception as e:
                print(f"    ⚠️  跳過（{e}）")

    print(f"\n✅ 搜尋完成！資料已儲存至 '{output_root}' 資料夾。")


def mode_place(args):
    """Fetch info and images for one specific place."""
    if args.place_id:
        place_id = args.place_id
        # Resolve name and location from the place ID
        basic = gmaps.place(place_id=place_id, fields=['name', 'geometry'], language='zh-TW')['result']
        name = basic.get('name', place_id)
        dest = basic['geometry']['location']
    else:
        # Find place by name
        found = gmaps.find_place(
            input=args.name,
            input_type='textquery',
            fields=['place_id', 'name', 'geometry'],
            language='zh-TW'
        )
        candidates = found.get('candidates', [])
        if not candidates:
            print(f"❌ 找不到地點：{args.name}")
            return
        candidate = candidates[0]
        place_id = candidate['place_id']
        name = candidate.get('name', args.name)
        dest = candidate['geometry']['location']
        print(f"找到地點：{name}（place_id: {place_id}）")

    output_root = args.output or re.sub(r'[\\/*?:"<>| ]', "_", name) + "_place"
    os.makedirs(output_root, exist_ok=True)
    csv_path = os.path.join(output_root, "data.csv")

    with open(csv_path, mode='w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()
        fetch_details_and_save(
            place_id=place_id,
            name=name,
            dest=dest,
            origin=None,
            output_root=output_root,
            folder_prefix="1",
            writer=writer
        )

    print(f"\n✅ 完成！資料已儲存至 '{output_root}' 資料夾。")


def main():
    parser = argparse.ArgumentParser(description='Google Maps 資料抓取工具')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # --- search mode ---
    sp = subparsers.add_parser('search', help='搜尋附近地點')
    sp.add_argument('--keyword', required=True, help='搜尋關鍵字，例如：咖啡廳、餐廳、飯店')
    sp.add_argument('--lat', type=float, default=25.0720, help='起點緯度（預設：25.0720）')
    sp.add_argument('--lng', type=float, default=121.5205, help='起點經度（預設：121.5205）')
    sp.add_argument('--radius', type=int, default=1000, help='搜尋半徑（公尺，預設：1000）')
    sp.add_argument('--limit', type=int, default=10, help='最多取幾筆結果（預設：10）')
    sp.add_argument('--output', help='輸出資料夾名稱（預設依關鍵字自動命名）')

    # --- place mode ---
    pp = subparsers.add_parser('place', help='抓取特定地點的資訊與圖片')
    group = pp.add_mutually_exclusive_group(required=True)
    group.add_argument('--name', help='地點名稱，例如：鼎泰豐信義店')
    group.add_argument('--place-id', dest='place_id', help='Google Maps Place ID')
    pp.add_argument('--output', help='輸出資料夾名稱（預設依地點名稱自動命名）')

    args = parser.parse_args()
    if args.mode == 'search':
        mode_search(args)
    elif args.mode == 'place':
        mode_place(args)


if __name__ == '__main__':
    main()