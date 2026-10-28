# Quy tắc workspace cho Antigravity / agy

Đọc và tuân thủ toàn bộ [AGENTS.md](AGENTS.md) trước khi làm việc trong repo; đây là nguồn quy tắc chung. Khi dùng Kaggle, đọc `AGENTS.local.md` nếu tồn tại, không đưa credentials/username thật vào file tracked.

Trước mỗi đợt sửa hoặc thay thế plan đang hiệu lực:

1. Lưu nguyên văn bản hiện tại trên ổ đĩa vào `.vscode/history_plan/<track>/YYYY-MM-DD_HHMMSS+0700__<tên-file>__<lý-do>.md` và kiểm SHA-256 khớp preimage.
2. Cập nhật `.vscode/history_plan/INDEX.md` với nguồn, thời điểm, lý do, checksum và bản kế tiếp; chỉ sau đó mới sửa plan.
3. Giữ snapshot bất biến và thêm liên kết lịch sử vào plan mới. Không lấy HEAD/staging thay cho nội dung chưa commit trên ổ đĩa; không tự commit/push.

Chi tiết và các ngoại lệ về run card/INDEX nằm trong mục **Lưu lịch sử plan trước khi sửa** của AGENTS.md và [hướng dẫn archive](.vscode/history_plan/README.md).
