# TTS cho LifeMate — 3 nguồn

| File | Cách dùng |
|---|---|
| `colab/lifemate_vieneu_colab.ipynb` | Colab **T4 GPU**, thử đọc tiếng Việt bằng giọng có sẵn |
| `colab/Omnivoice.ipynb` | Colab **T4 GPU**, Auto Voice tiếng Việt, không cần audio tham chiếu |
| `deepgram/run_deepgram.py` | Chạy trên máy, hai giọng Flux Hoa và Minh |
| `colab/record_my_voice.html` | Thu giọng của bạn để dùng sau (tùy chọn) |

Hai notebook hiện chỉ để **test text → speech**. Mỗi notebook có 4 cell: cài thư viện → nhập `TEXTS` → tải model GPU → tạo WAV và nghe thử. Upload lên Colab, chọn **T4 GPU**, rồi **Run all**; dùng phiên Colab riêng cho từng notebook. Chưa có bước đọc dataset; khi có dữ liệu sẽ thêm đường dẫn sau. WAV thử được lưu ở `/content/vieneu_test` hoặc `/content/omnivoice_test`, giữ sample rate gốc của model.

## Deepgram trên máy

Từ thư mục `synthetic-data-tts`:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python deepgram/run_deepgram.py --dry-run
.venv/bin/python deepgram/run_deepgram.py --limit 1
```

Key có sẵn trong `deepgram/.env`, bị Git bỏ qua. Mặc định gọi `flux-hoa-vi` và `flux-minh-vi` qua API v2. Mỗi câu đọc bằng hai giọng được tính thành hai lượt tổng hợp. `tts_common.py` là module cần cho script Deepgram, không cần mở/sửa khi chạy.

Khi có dữ liệu, tạo JSONL UTF-8 với `id` duy nhất và `text`:

```json
{"id":"syn_0001","text":"Nhắn cho mẹ là tối nay con về muộn."}
```

Thêm `text_tts` nếu cần viết số/ngày giờ thành chữ, phải giữ nguyên ý định. Chạy `--input /duong/dan/texts.jsonl --all --dry-run` để kiểm trước; bỏ `--dry-run` để tạo audio. Output ở `deepgram/outputs/`, gồm WAV mono PCM16 16 kHz và manifest. Chạy lại bỏ qua file hợp lệ; chỉ một tiến trình ghi một output.

Thu giọng: mở `colab/record_my_voice.html`, thu 3–8 giây, nghe lại và tải audio + JSON lời đọc. Nếu cần localhost, chạy `python3 -m http.server 8765 --bind 127.0.0.1` trong `colab`, rồi mở `http://127.0.0.1:8765/record_my_voice.html`.

Auto Voice là giọng ngẫu nhiên; cùng seed không bảo đảm cùng danh tính giọng giữa nhiều câu. Nghe kiểm tên riêng, số tiền, ngày giờ trước ASR/SFT. Bộ TTS không sinh trajectory gold.
