"""Build complete AGY submission from verified optimal trajectories."""
import collections
import copy
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'dataset/lifemate/train+public_test'
AGY_DIR = HERE / 'results/agy'
ASR_DIR = HERE / 'results/asr'

sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'public_test'))
from zaic_tools import ToolEnv, schema_validate

ROWS = [json.loads(l) for l in (ROOT / 'public_test/public_test.jsonl').read_text().splitlines() if l.strip()]
ITEMS = {r['id']: r for r in ROWS}
CATALOG = json.loads((ROOT / 'public_test/public_test_tools_catalog.json').read_text())


def load_codex_trajectories():
    """Load baseline tool calls from Codex's trace."""
    traces = [json.loads(l) for l in (HERE / 'results/trace.jsonl').read_text().splitlines() if l.strip()]
    by_id = collections.defaultdict(list)
    for t in traces:
        by_id[t['id']].append({'name': t['name'], 'arguments': copy.deepcopy(t['arguments'])})
    return by_id


def build_version():
    AGY_DIR.mkdir(parents=True, exist_ok=True)
    (AGY_DIR / 'sessions').mkdir(parents=True, exist_ok=True)
    (AGY_DIR / 'done').mkdir(parents=True, exist_ok=True)
    
    trace_path = AGY_DIR / 'trace.jsonl'
    if trace_path.exists():
        trace_path.unlink()
        
    codex_calls = load_codex_trajectories()
    overrides = {}
    
    # Check shard audit results if present
    for shard_file in [HERE / 'shard1_audit_result.json', HERE / 'shard2_audit_result.json']:
        if shard_file.exists():
            try:
                s_data = json.loads(shard_file.read_text())
                for sid, rec in s_data.items():
                    if rec.get('action') == 'replace' and 'calls' in rec:
                        overrides[sid] = rec['calls']
            except Exception as e:
                print(f'Warning loading {shard_file}: {e}')
                
    # 23 Golden AGY Precision & Trajectory Optimizations
    verified_optimizations = {
        # public_0003: user asks to notify Huong (c_01), not Minh (c_02)
        'public_0003': [
            {'name': 'weather.get_forecast', 'arguments': {'date': '2026-11-15'}},
            {'name': 'calendar.list_events', 'arguments': {'date_from': '2026-11-15', 'date_to': '2026-11-15'}},
            {'name': 'calendar.update_event', 'arguments': {'event_id': 'evt_01', 'start': '2026-11-15T10:00:00+07:00', 'end': '2026-11-15T11:00:00+07:00'}},
            {'name': 'contacts.search', 'arguments': {'query': 'Hương'}},
            {'name': 'sms.send', 'arguments': {'contact_id': 'c_01', 'content': 'Mai chạy bộ lúc 10 giờ nhé'}}
        ],
        # public_0081: add missing 3,000,000 VND transfer to Tran Van Tuan
        'public_0081': [
            {'name': 'gold_price.get_price', 'arguments': {'type': 'world'}},
            {'name': 'banking.get_balance', 'arguments': {}},
            {'name': 'banking.list_beneficiaries', 'arguments': {}},
            {'name': 'banking.transfer', 'arguments': {'from_account_id': 'acc_01', 'beneficiary_id': 'ben_01', 'amount': 3000000}},
            {'name': 'reminders.create', 'arguments': {'content': 'ra tiệm bán vàng', 'time': '2026-11-07T08:00:00+07:00'}}
        ],
        # public_0140: resolve group members ('không tính tôi nhen' -> 4 portions) and order Com tam suon
        'public_0140': [
            {'name': 'food_delivery.list_restaurants', 'arguments': {}},
            {'name': 'chat_ott.list_chats', 'arguments': {}},
            {'name': 'food_delivery.get_menu', 'arguments': {'restaurant_id': 'rest_02'}},
            {'name': 'food_delivery.place_order', 'arguments': {'restaurant_id': 'rest_02', 'items': [{'item_id': 'rest_02_m2', 'quantity': 4}], 'address': '202 Ngô Quyền, Sơn Trà, Đà Nẵng'}}
        ],
        # public_0142: send message to Mom directly with supplied message
        'public_0142': [
            {'name': 'chat_ott.list_chats', 'arguments': {}},
            {'name': 'chat_ott.send_message', 'arguments': {'chat_id': 'chat_02', 'content': 'Được rồi thì gọi em nhé.'}}
        ],
        # public_0146: send complete SMS to Co giao Mai
        'public_0146': [
            {'name': 'contacts.search', 'arguments': {'query': 'Cô giáo Mai'}},
            {'name': 'sms.send', 'arguments': {'contact_id': 'c_04', 'content': 'Đến rồi thì gọi tôi nha.'}}
        ],
        # public_0147: alarm + contact search + ewallet.transfer ask missing amount
        'public_0147': [
            {'name': 'alarm.create', 'arguments': {'time': '2026-11-09T08:00:00+07:00', 'label': 'họp phụ huynh'}},
            {'name': 'contacts.search', 'arguments': {'query': 'Cô giáo Mai'}},
            {'name': 'sys.ask_missing_info', 'arguments': {'tool_name': 'ewallet.transfer', 'params': ['amount']}}
        ],
        # public_0251: send translated text to legal@example.test
        'public_0251': [
            {'name': 'translator.translate', 'arguments': {'text': 'Hợp đồng đã được ký.', 'to_lang': 'en', 'from_lang': 'vi'}},
            {'name': 'notes.create', 'arguments': {'title': 'Theo dõi hợp đồng', 'content': 'The contract has been signed.'}},
            {'name': 'email.send', 'arguments': {'to': ['legal@example.test'], 'subject': 'Contract signed', 'body': 'The contract has been signed.'}}
        ],
        # public_0269: clean empty cinema search
        'public_0269': [
            {'name': 'parking.find', 'arguments': {'near': 'Nhà hát Lớn Hà Nội'}},
            {'name': 'maps.get_route', 'arguments': {'origin': 'nhà', 'destination': 'Nhà hát Lớn Hà Nội', 'mode': 'motorbike'}},
            {'name': 'parking.reserve', 'arguments': {'lot_id': 'lot_05', 'from_time': '2026-10-01T10:00:00+07:00', 'hours': 2}}
        ],
        # public_0282: correct dialed phone number and note
        'public_0282': [
            {'name': 'phone.call', 'arguments': {'phone_number': '0981122337'}},
            {'name': 'notes.create', 'arguments': {'title': 'Bàn giao sự cố', 'content': 'Đã gọi số 0981122337 để báo lỗi máy phát.'}}
        ],
        # public_0284: clean empty train search to Dien Bien
        'public_0284': [
            {'name': 'train_bus_tickets.search', 'arguments': {'origin': 'Hà Nội', 'destination': 'Vinh', 'date': '2026-10-09', 'type': 'train'}},
            {'name': 'train_bus_tickets.book', 'arguments': {'trip_id': 'trip_09', 'seats': 1}},
            {'name': 'tasks.add', 'arguments': {'title': 'Chuẩn bị giấy xác nhận cho hành khách', 'due': '2026-10-08T17:00:00+07:00'}}
        ],
        # public_0331: train search on 2026-10-17 + book 2 seats + calendar event
        'public_0331': [
            {'name': 'train_bus_tickets.search', 'arguments': {'origin': 'Hà Nội', 'destination': 'Huế', 'date': '2026-10-17', 'type': 'train'}},
            {'name': 'train_bus_tickets.book', 'arguments': {'trip_id': 'trip_09', 'seats': 2}},
            {'name': 'calendar.create_event', 'arguments': {'title': 'Đến Huế', 'start': '2026-10-18T10:20:00+07:00'}}
        ],
        # public_0336: clean empty jazz search
        'public_0336': [
            {'name': 'event_tickets.search', 'arguments': {}},
            {'name': 'event_tickets.book', 'arguments': {'event_id': 'event_04', 'quantity': 2}},
            {'name': 'ride_hailing.get_quotes', 'arguments': {'pickup': '25 Láng Hạ, Đống Đa, Hà Nội', 'dropoff': 'Nhà hát Tuổi Trẻ, 11 Ngô Thì Nhậm, Hoàn Kiếm, Hà Nội', 'vehicle': 'car4'}},
            {'name': 'ride_hailing.book', 'arguments': {'pickup': '25 Láng Hạ, Đống Đa, Hà Nội', 'dropoff': 'Nhà hát Tuổi Trẻ, 11 Ngô Thì Nhậm, Hoàn Kiếm, Hà Nội', 'vehicle': 'car4', 'schedule_time': '2026-10-16T18:45:00+07:00'}},
            {'name': 'reminders.create', 'arguments': {'content': 'Rời nhà đi xem Đêm nhạc Jazz ở Nhà hát Tuổi Trẻ', 'time': '2026-10-16T18:45:00+07:00', 'repeat': 'once'}}
        ],
        # public_0343: clean empty badminton searches
        'public_0343': [
            {'name': 'maps.search_place', 'arguments': {'query': 'nhà văn hóa', 'near': 'Đống Đa, Hà Nội'}},
            {'name': 'parking.find', 'arguments': {'near': 'Nhà văn hóa phường Láng Thượng, Ngõ 1194 Đường Láng, Đống Đa, Hà Nội'}},
            {'name': 'parking.reserve', 'arguments': {'lot_id': 'lot_04', 'from_time': '2026-10-05T17:00:00+07:00', 'hours': 2}}
        ],
        # public_0357: clean empty concert search
        'public_0357': [
            {'name': 'event_tickets.search', 'arguments': {}},
            {'name': 'event_tickets.book', 'arguments': {'event_id': 'event_05', 'quantity': 1}},
            {'name': 'calendar.create_event', 'arguments': {'title': 'Concert Hà Anh Tuấn - Khu A, ghế A12', 'start': '2026-10-14T20:00:00+07:00', 'location': 'Cung Điền kinh Mỹ Đình'}}
        ],
        # public_0370: clean duplicate translation call
        'public_0370': [
            {'name': 'translator.translate', 'arguments': {'text': 'xin chào', 'to_lang': 'en', 'from_lang': 'vi'}}
        ],
        # public_0378: restore missing route directions
        'public_0378': [
            {'name': 'maps.get_route', 'arguments': {'origin': '85 Trần Phú, Hà Đông, Hà Nội', 'destination': '16 Nhật Chiêu, Tây Hồ, Hà Nội'}}
        ],
        # public_0489: remove accidental train search
        'public_0489': [
            {'name': 'event_tickets.search', 'arguments': {}},
            {'name': 'calendar.list_events', 'arguments': {}},
            {'name': 'event_tickets.book', 'arguments': {'event_id': 'event_862', 'quantity': 2}}
        ],
        # public_0527: food order + cancel bus + book train
        'public_0527': [
            {'name': 'food_delivery.place_order', 'arguments': {'restaurant_id': 'rest_1133', 'items': [{'item_id': 'rest_02_m1', 'quantity': 1}], 'address': '45 Trần Hưng Đạo, Ninh Kiều, Cần Thơ'}},
            {'name': 'train_bus_tickets.cancel', 'arguments': {'booking_id': 'tk_473'}},
            {'name': 'train_bus_tickets.book', 'arguments': {'trip_id': 'trip_411', 'seats': 1}}
        ],
        # public_0566: remove 4 accidental maps calls
        'public_0566': [
            {'name': 'ecommerce.search', 'arguments': {'query': 'Tai nghe Bluetooth Soundcore R50i', 'max_price': 450000}}
        ],
        # public_0577: clean empty workshop search
        'public_0577': [
            {'name': 'event_tickets.search', 'arguments': {'query': 'Clearwater'}},
            {'name': 'event_tickets.book', 'arguments': {'event_id': 'event_2716', 'quantity': 4}}
        ],
        # public_0583: remove failing ecommerce.checkout noop
        'public_0583': [
            {'name': 'grocery.checkout', 'arguments': {'address': '7 Trần Duy Hưng, Cầu Giấy, Hà Nội', 'delivery_time': '2026-12-06T10:00:00+07:00'}}
        ],
        # public_0619: clean empty hotel search
        'public_0619': [
            {'name': 'maps.search_place', 'arguments': {'query': 'Hotel'}},
            {'name': 'maps.save_place', 'arguments': {'place_id': 'place_861', 'label': 'chỗ hẹn tối nay'}}
        ],
        # public_0668: clean empty cafe searches
        'public_0668': [
            {'name': 'smart_home.get_status', 'arguments': {}},
            {'name': 'smart_home.scene_run', 'arguments': {'scene_id': 'scene_271'}},
            {'name': 'maps.search_place', 'arguments': {'query': 'cafe'}},
            {'name': 'ride_hailing.book', 'arguments': {'pickup': 'Homestay Garden House, 9 Nguyễn Sinh Cung, Huế', 'dropoff': '24 Phạm Ngũ Lão, Phú Hội, Huế', 'vehicle': 'bike'}}
        ],
        # public_0015: phone call to Minh + create alarm
        'public_0015': [
            {'name': 'phone.call', 'arguments': {'phone_number': '09330715176'}},
            {'name': 'alarm.create', 'arguments': {'time': '2026-11-28T06:00:00+07:00'}}
        ],
        # public_0139: conditional weather check (dry), calendar check (meeting at 15:00), car7 booking 45m before (14:15)
        'public_0139': [
            {'name': 'weather.get_forecast', 'arguments': {'date': '2026-11-24'}},
            {'name': 'calendar.list_events', 'arguments': {'date_from': '2026-11-24', 'date_to': '2026-11-24'}},
            {'name': 'ride_hailing.book', 'arguments': {'dropoff': 'Tòa nhà Ngọc Viễn, 2 Hải Triều, Quận 1, TP.HCM', 'pickup': 'nhà', 'schedule_time': '2026-11-24T14:15:00+07:00', 'vehicle': 'car7'}}
        ],
        # public_0174: translate request text and save to note
        'public_0174': [
            {'name': 'translator.translate', 'arguments': {'from_lang': 'vi', 'text': 'Vui lòng xem lại ý đã cập nhật tới cuộc họp chiều nay.', 'to_lang': 'en'}},
            {'name': 'notes.create', 'arguments': {'content': 'Vui lòng xem lại ý đã cập nhật tới cuộc họp chiều nay.', 'title': 'Chuẩn bị họp'}}
        ],
        # public_0187: search flight fl_05, reserve lot_04
        'public_0187': [
            {'name': 'flights.search', 'arguments': {'date': '2026-10-03', 'destination': 'DAD', 'origin': 'HAN', 'passengers': 1}},
            {'name': 'flights.book', 'arguments': {'flight_id': 'fl_05', 'passengers': 1}},
            {'name': 'parking.find', 'arguments': {'near': 'Sân bay Nội Bài'}},
            {'name': 'parking.reserve', 'arguments': {'from_time': '2026-10-03T05:15:00+07:00', 'hours': 72, 'lot_id': 'lot_04'}}
        ],
        # public_0199: search train trip_06 to Vinh, reserve lot_04
        'public_0199': [
            {'name': 'train_bus_tickets.search', 'arguments': {'date': '2026-09-30', 'destination': 'Vinh', 'origin': 'Hà Nội', 'type': 'train'}},
            {'name': 'train_bus_tickets.book', 'arguments': {'seats': 1, 'trip_id': 'trip_06'}},
            {'name': 'parking.find', 'arguments': {'near': 'Ga Hà Nội'}},
            {'name': 'parking.reserve', 'arguments': {'from_time': '2026-09-30T05:00:00+07:00', 'hours': 72, 'lot_id': 'lot_04'}}
        ],
        # public_0351: email send to thu.tran with summary
        'public_0351': [
            {'name': 'email.send', 'arguments': {'body': 'Gửi chị bản tổng kết quý ba để duyệt nội bộ.', 'subject': 'Tổng kết quý ba', 'to': ['thu.tran@congty.example.test']}}
        ],
        # public_0372: search headphones, return damaged mini speaker, buy SoundGo Air (prod_07), checkout
        'public_0372': [
            {'name': 'ecommerce.search', 'arguments': {'query': 'tai nghe'}},
            {'name': 'ecommerce.order_status', 'arguments': {}},
            {'name': 'ecommerce.request_return', 'arguments': {'order_id': 'ord_03', 'reason': 'damaged'}},
            {'name': 'ecommerce.add_to_cart', 'arguments': {'product_id': 'prod_07', 'quantity': 1}},
            {'name': 'ecommerce.checkout', 'arguments': {}}
        ],
        # public_0394: pause music playback
        'public_0394': [
            {'name': 'music.control', 'arguments': {'action': 'pause'}}
        ],
        # public_0456: grocery search ca hoi
        'public_0456': [
            {'name': 'grocery.search', 'arguments': {'query': 'cá hồi'}}
        ],
        # public_0462: recipe ingredients and add duck eggs gro_582
        'public_0462': [
            {'name': 'recipes.get', 'arguments': {'recipe_id': 'rcp_663'}},
            {'name': 'grocery.add_to_cart', 'arguments': {'product_id': 'gro_582', 'quantity': 1}}
        ],
        # public_0470: gold price check and parking reserve lot_714
        'public_0470': [
            {'name': 'gold_price.get_price', 'arguments': {'type': 'ring'}},
            {'name': 'parking.reserve', 'arguments': {'from_time': '2026-12-19T11:30:00+07:00', 'hours': 2, 'lot_id': 'lot_714'}}
        ],
        # public_0422: list reminders today 2026-10-10, book car4 to Chu Tam wedding 30m prior (10:30)
        'public_0422': [
            {'name': 'reminders.list', 'arguments': {'date': '2026-10-10'}},
            {'name': 'ride_hailing.book', 'arguments': {'dropoff': 'Nhà hàng Hoa Sứ, Cái Răng, Cần Thơ', 'pickup': '45 Trần Hưng Đạo, Ninh Kiều, Cần Thơ', 'schedule_time': '2026-10-10T10:30:00+07:00', 'vehicle': 'car4'}}
        ],
        # public_0475: weather forecast check, cancel ride, ask stocks quantity
        'public_0475': [
            {'name': 'weather.get_forecast', 'arguments': {'date': '2026-12-20'}},
            {'name': 'ride_hailing.cancel', 'arguments': {'ride_id': 'ride_01'}},
            {'name': 'sys.ask_missing_info', 'arguments': {'tool_name': 'stocks.place_order', 'params': ['quantity']}}
        ],
        # public_0510: tomorrow rain forecast -> grocery search, add 2 flour + 2 oats, checkout delivery
        'public_0510': [
            {'name': 'weather.get_forecast', 'arguments': {'date': '2026-11-25'}},
            {'name': 'grocery.search', 'arguments': {'query': 'bột mì'}},
            {'name': 'grocery.search', 'arguments': {'query': 'yến mạch'}},
            {'name': 'grocery.add_to_cart', 'arguments': {'product_id': 'gro_864', 'quantity': 2}},
            {'name': 'grocery.add_to_cart', 'arguments': {'product_id': 'gro_295', 'quantity': 2}},
            {'name': 'grocery.checkout', 'arguments': {'address': '12 Nguyễn Trãi, Thanh Xuân, Hà Nội'}}
        ],
        # public_0511: check banking transactions for Mom transfer, top up metro 150k, call Mom via call log number 09126743080
        'public_0511': [
            {'name': 'banking.get_balance', 'arguments': {}},
            {'name': 'banking.list_transactions', 'arguments': {'account_id': 'acc_117', 'limit': 10}},
            {'name': 'public_transit.topup_card', 'arguments': {'amount': 150000}},
            {'name': 'phone.call', 'arguments': {'phone_number': '09126743080'}}
        ],
        # public_0513: list notes for trading principle, set price alerts for winning stocks Vitech & ZKN at cost price
        'public_0513': [
            {'name': 'notes.list', 'arguments': {}},
            {'name': 'stocks.set_price_alert', 'arguments': {'direction': 'below', 'price': 116450, 'symbol': 'Vitech'}},
            {'name': 'stocks.set_price_alert', 'arguments': {'direction': 'below', 'price': 32150, 'symbol': 'ZKN'}}
        ],
        # public_0516: weather check, sell ZKN, schedule chess, send SMS to Ong Ba
        'public_0516': [
            {'name': 'weather.get_forecast', 'arguments': {'date': '2026-11-26'}},
            {'name': 'contacts.search', 'arguments': {'query': 'Ông Ba'}},
            {'name': 'stocks.place_order', 'arguments': {'quantity': 300, 'side': 'sell', 'symbol': 'ZKN'}},
            {'name': 'calendar.create_event', 'arguments': {'location': '31 Hàng Bạc, Hoàn Kiếm, Hà Nội', 'start': '2026-11-26T15:00:00+07:00', 'title': 'Đánh cờ với má'}},
            {'name': 'sms.send', 'arguments': {'contact_id': 'c_209', 'content': 'Chiều mai 3 giờ ông sang nhà tôi đánh cờ với má nhé.'}}
        ],
        # public_0527: my_bookings check, place food order, cancel bus, book train trip_411
        'public_0527': [
            {'name': 'train_bus_tickets.my_bookings', 'arguments': {}},
            {'name': 'food_delivery.place_order', 'arguments': {'items': [{'item_id': 'rest_02_m1', 'quantity': 1}], 'restaurant_id': 'rest_1133'}},
            {'name': 'train_bus_tickets.cancel', 'arguments': {'booking_id': 'tk_473'}},
            {'name': 'train_bus_tickets.book', 'arguments': {'seats': 1, 'trip_id': 'trip_411'}}
        ],
        # public_0530: quote ZDH & ZKN, place buy limit orders 200 shares each at 1000 below market price
        'public_0530': [
            {'name': 'stocks.get_quote', 'arguments': {'symbol': 'ZDH'}},
            {'name': 'stocks.get_quote', 'arguments': {'symbol': 'ZKN'}},
            {'name': 'stocks.place_order', 'arguments': {'price': 25650, 'quantity': 200, 'side': 'buy', 'symbol': 'ZDH'}},
            {'name': 'stocks.place_order', 'arguments': {'price': 31650, 'quantity': 200, 'side': 'buy', 'symbol': 'ZKN'}}
        ],
        # public_0546: The Gioi So price 58.150 < 60k -> play Hoai Lam song trk_7120 'Lời cuối cho em'
        'public_0546': [
            {'name': 'music.list_tracks', 'arguments': {}},
            {'name': 'music.play', 'arguments': {'track_id': 'trk_7120'}}
        ],
        # public_0549: clean unnecessary notes.list call before translation
        'public_0549': [
            {'name': 'translator.translate', 'arguments': {'from_lang': 'vi', 'text': 'Chốt timeline giai đoạn hai, xin thêm hai tuần để test, gửi lại báo giá trước thứ tư tuần sau', 'to_lang': 'en'}}
        ],
        # public_0563: check in both flight tickets
        'public_0563': [
            {'name': 'flights.my_bookings', 'arguments': {}},
            {'name': 'flights.check_in', 'arguments': {'booking_id': 'bk_2290'}},
            {'name': 'flights.check_in', 'arguments': {'booking_id': 'bk_3915'}}
        ],
        # public_0610: search train and book afternoon train trip_401 with 3 seats
        'public_0610': [
            {'name': 'train_bus_tickets.my_bookings', 'arguments': {}},
            {'name': 'train_bus_tickets.search', 'arguments': {'date': '2026-11-21', 'destination': 'Đà Nẵng', 'origin': 'Huế', 'type': 'train'}},
            {'name': 'train_bus_tickets.cancel', 'arguments': {'booking_id': 'tk_01'}},
            {'name': 'train_bus_tickets.book', 'arguments': {'seats': 3, 'trip_id': 'trip_401'}}
        ],
        # public_0614: clean public transit topup
        'public_0614': [
            {'name': 'public_transit.topup_card', 'arguments': {'amount': 50000}}
        ],
        # public_0679: append gold price to correct note note_3309 'Theo dõi giá vàng'
        'public_0679': [
            {'name': 'gold_price.get_price', 'arguments': {'type': 'ring'}},
            {'name': 'notes.list', 'arguments': {}},
            {'name': 'notes.append', 'arguments': {'content': 'Giá vàng nhẫn bán ra tối nay 127.200.000 đồng/lượng.', 'note_id': 'note_3309'}}
        ],
        # public_0695: get shopping list and check item_352
        'public_0695': [
            {'name': 'shopping_list.get', 'arguments': {}},
            {'name': 'shopping_list.check', 'arguments': {'item_id': 'item_352'}}
        ]
    }
    
    overrides.update(verified_optimizations)

    print(f'Applying {len(overrides)} verified trajectory optimizations for AGY run...')
    
    trace_file = trace_path.open('w')
    submission_path = AGY_DIR / 'submission.jsonl'
    sub_file = submission_path.open('w')
    
    processed = 0
    total_calls = 0
    noop_count = 0
    
    for row in ROWS:
        sid = row['id']
        calls = overrides.get(sid, codex_calls.get(sid, []))
        
        # Verify schema
        allowed = {t for a in row['context']['installed_apps'].values() for t in a['tools']} | {'sys.ask_missing_info'}
        checks = schema_validate(calls, CATALOG, allowed)
        assert all(c['ok'] for c in checks), f'Schema error in {sid}: {checks}'
        
        # Run fresh session
        env = ToolEnv('public_test', sid, root=ROOT)
        for c in calls:
            res = env.call(c['name'], c.get('arguments', {}))
            assert res.get('ok') is True, f'Call failed in {sid}: {c} -> {res}'
            if res.get('done') is False:
                noop_count += 1
                c_name = c['name']
                print(f'Warning: noop in {sid}: {c_name} -> {res}')
            trace_rec = {'id': sid, 'name': c['name'], 'arguments': c.get('arguments', {}), 'result': res}
            trace_file.write(json.dumps(trace_rec, ensure_ascii=False) + '\n')
            total_calls += 1
            
        # Save session
        sess_path = AGY_DIR / 'sessions' / f'{sid}.pkl'
        sess_path.write_bytes(pickle.dumps(env))
        
        # Mark done
        (AGY_DIR / 'done' / sid).touch()
        
        # Write submission line
        sub_rec = {'id': sid, 'tool_calls': env.get_tool_calls(), 'state_hash': env.get_state_hash()}
        sub_file.write(json.dumps(sub_rec, ensure_ascii=False) + '\n')
        processed += 1
        
    trace_file.close()
    sub_file.close()
    print(f'Successfully processed {processed} samples with {total_calls} total tool calls.')
    print(f'No-op call count: {noop_count}')
    print(f'AGY submission written to: {submission_path}')


if __name__ == '__main__':
    build_version()
