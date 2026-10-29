# Tracking

Mỗi track **một file duy nhất**: [`lifemate.md`](lifemate.md), [`datamate.md`](datamate.md). Mỗi file gồm:

- **Runs**: một dòng mỗi lần chạy Kaggle (kể cả hỏng hoặc mới dự định): `run_id`, account bí danh `acc<N>`, status, thay đổi chính, kết quả, kết luận.
- **Nộp Public LB**: một dòng mỗi lần nộp (tối đa 5 lần/ngày/track).
- **Đã xác nhận / Ngõ cụt / Nhóm lỗi / Chưa kiểm**: kiến thức rút ra, mỗi mục một dòng có ngày + nguồn.

## Quy ước

- `run_id`: `<lm|dm>-<lane>-<NNN>-<mô-tả-ngắn>`, NNN tăng dần theo track, không đổi sau khi tạo. Mỗi run một thay đổi chính so với run trước để so được.
- Bắt đầu run: thêm dòng status `running`. Xong: điền kết quả (điểm dev, s/mẫu, peak VRAM nếu có) và đổi status `done` / `failed`. Run hỏng vẫn giữ dòng để không chạy lại cùng cấu hình.
- Username Kaggle thật chỉ nằm trong `AGENTS.local.md`.

## Không làm

- **Không tạo file mới** trong thư mục này: không run card riêng, không plan riêng, không snapshot, không SHA-256.
- Không ghi sửa vặt (typo, link, chỉnh câu chữ).
- Bản cũ xem bằng `git log -p <file>`.
