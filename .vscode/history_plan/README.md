# Lịch sử phiên bản kế hoạch

`planning-research/<track>/planning.md` là plan đang hiệu lực. Thư mục này giữ nguyên văn các bản trước mỗi đợt sửa/thay thế để truy vết quyết định; không dùng snapshot cũ làm plan hiện hành.

## Quy trình bắt buộc

1. Đọc quy tắc chung trong `../../AGENTS.md`, plan hiện hành và `INDEX.md` ở đây trước khi sửa.
2. Đọc/capture nguyên văn file hiện tại trên ổ đĩa, tính SHA-256. Lưu bản đó vào `<track>/YYYY-MM-DD_HHMMSS+0700__<tên-file>__<lý-do>.md` **trước khi sửa**. Dùng chế độ tạo file mới, không ghi đè nếu trùng tên; thêm hậu tố duy nhất khi cần. Nếu thay thế toàn bộ plan thì chuyển bản cũ vào lịch sử rồi tạo bản hiện hành mới; nếu chỉnh một phần thì snapshot đầy đủ trước khi edit.
3. Đối chiếu bytes/SHA-256 snapshot với preimage; chỉ tiếp tục khi khớp. Một snapshot cho mỗi file trong một đợt chỉnh sửa liên tục; nếu agent khác sửa file trong lúc đó, đọc lại và lưu preimage mới trước khi tiếp tục.
4. Thêm dòng `INDEX.md`: thời điểm lưu có múi giờ, track, đường dẫn nguồn, snapshot, SHA-256, lý do, nguồn phục hồi nếu có và bản kế tiếp. Plan hiện hành có liên kết ngược về snapshot/INDEX.
5. Snapshot là bất biến: không chỉnh nội dung, không xóa, không tái sử dụng tên file. Mọi đính chính về nguồn/thời điểm ghi bổ sung trong INDEX. Không cần snapshot đệ quy cho chính INDEX/README hoặc cập nhật vận hành run card/insights.
6. Không dùng `git show HEAD`/staging thay cho bản trên ổ đĩa trong quy trình bình thường: plan có thể chứa thay đổi chưa commit của người dùng/agent khác. Không sửa Git index để lưu lịch sử; không tự commit/push.

## Nguồn của các bản khôi phục đầu tiên

Các snapshot `recovered-before-baseline-dev` được tạo **sau** lượt sửa plan ngày 2026-10-29 để khôi phục thiếu sót lưu lịch sử. Bytes lấy từ Git index, được đối chiếu khớp với bản trước lượt sửa (bỏ đúng phần thêm mục 9–10 và thay đổi câu trạng thái). Thời điểm trong tên là lúc khôi phục/lưu archive, không phải thời điểm plan gốc được viết. Không suy diễn rằng các phiên bản cũ hơn cũng đã được khôi phục.

Các snapshot `before-history-policy` là nguyên văn working tree trước khi bổ sung liên kết lịch sử trong lượt khắc phục này. Chúng giữ lại bản kế hoạch đã được duyệt ở lượt trước.

Snapshot cố ý giữ nguyên byte, nên link tương đối bên trong được hiểu theo **đường dẫn nguồn ban đầu** ghi ở INDEX; không sửa các link đó ngay trong snapshot.

## Entry point cho các agent

- Codex: quy tắc chính nằm trong `../../AGENTS.md`.
- Claude Code: `../../CLAUDE.md` import AGENTS.md và nhắc quy tắc trước-edit.
- Antigravity/agy: `../../GEMINI.md` dẫn vào AGENTS.md và yêu cầu lưu lịch sử trước-edit. Cách đặt AGENTS.md/GEMINI.md tại workspace được mô tả trong [tài liệu Rules của Antigravity](https://antigravity.google/docs/rules/).

Đây là hướng dẫn ở cấp repo; chưa kiểm chứng hành vi thực tế của từng ứng dụng đang chạy. Không chỉnh cấu hình toàn máy.
