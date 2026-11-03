"""Generic temporal arithmetic tests; no competition IDs, labels or state files."""
import copy
import importlib.util
import unittest
from pathlib import Path


class ActionNormalizerTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parent / "agent-action-fix/action_normalizer.py"
        self.assertTrue(path.exists(), "Action normalizer is missing")
        spec = importlib.util.spec_from_file_location("action_normalizer", path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.row = {"context": {"current_time": "2026-11-10T10:00:00+07:00"}, "history": []}
        self.events = [{"id": "event_fixture", "title": "Hẹn khám răng", "start": "2026-11-10T16:00:00+07:00",
                        "end": "2026-11-10T17:00:00+07:00"}]
        self.steps = [{"kind": "call", "name": "calendar.list_events", "arguments": {}, "result": {"events": self.events}}]

    def normalize(self, text, tool, args, steps=None):
        return self.mod.normalize_action(self.row, text, self.steps if steps is None else steps, tool, args)

    def test_eight_minus_fifteen_tomorrow_evening_is_1945(self):
        original = {"content": "gọi lại", "time": "2026-11-11T20:45:00+07:00"}
        fixed, audit = self.normalize("Nhắc tôi gọi lại lúc tám giờ kém mười lăm tối mai", "reminders.create", original)
        self.assertEqual(fixed["time"], "2026-11-11T19:45:00+07:00")
        self.assertEqual(original["time"], "2026-11-11T20:45:00+07:00")
        self.assertTrue(audit)

    def test_ride_pickup_and_reminder_use_different_offsets(self):
        text = "Chiều nay đặt ô tô tới chỗ hẹn khám răng trước giờ hẹn bốn mươi lăm phút, nhắc tôi mang laptop mười lăm phút trước lúc xe đón"
        fixed, _ = self.normalize(text, "ride_hailing.book", {"pickup": "A", "dropoff": "B", "vehicle": "car4"})
        self.assertEqual(fixed["schedule_time"], "2026-11-10T15:15:00+07:00")
        booked = self.steps + [{"kind": "call", "name": "ride_hailing.book", "arguments": fixed, "result": {"ok": True}}]
        reminder, _ = self.normalize(text, "reminders.create", {"content": "laptop", "time": "2026-11-10T16:00:00+07:00"}, booked)
        self.assertEqual(reminder["time"], "2026-11-10T15:00:00+07:00")

    def test_postpone_moves_both_endpoints_and_alarm_uses_updated_start(self):
        text = "Dời lịch khám răng chiều nay muộn sáu mươi phút, đặt báo thức trước đó một tiếng"
        fixed, _ = self.normalize(text, "calendar.update_event", {"event_id": "event_fixture", "start": "2026-11-10T15:00:00+07:00"})
        self.assertEqual(fixed["start"], "2026-11-10T17:00:00+07:00")
        self.assertEqual(fixed["end"], "2026-11-10T18:00:00+07:00")
        steps = self.steps + [{"kind": "call", "name": "calendar.update_event", "arguments": fixed, "result": {"ok": True}}]
        alarm, _ = self.normalize(text, "alarm.create", {"time": "2026-11-10T15:00:00+07:00"}, steps)
        self.assertEqual(alarm["time"], "2026-11-10T16:00:00+07:00")

    def test_does_not_guess_between_ambiguous_events_or_multiple_clocks(self):
        ambiguous = copy.deepcopy(self.steps)
        ambiguous[0]["result"]["events"].append(dict(self.events[0], id="other_fixture"))
        original = {"pickup": "A", "dropoff": "B", "vehicle": "car4", "schedule_time": "2026-11-10T15:50:00+07:00"}
        fixed, _ = self.normalize("Đặt xe tới lịch khám răng chiều nay trước giờ hẹn mười lăm phút", "ride_hailing.book", original, ambiguous)
        self.assertEqual(fixed, original)
        alarm = {"time": "2026-11-11T07:00:00+07:00"}
        fixed, _ = self.normalize("Báo thức bảy giờ sáng và nhắc thuốc lúc tám giờ tối mai", "alarm.create", alarm, [])
        self.assertEqual(fixed, alarm)

    def test_does_not_read_labels_or_change_unrelated_calls(self):
        self.row["label"] = {"tool_calls": [{"name": "alarm.create", "arguments": {"time": "SECRET"}}]}
        args = {"from_account_id": "a", "beneficiary_id": "b", "amount": 123}
        fixed, audit = self.normalize("Chuyển tiền trước tám giờ tối mai", "banking.transfer", args)
        self.assertEqual(fixed, args)
        self.assertEqual(audit, [])

    def test_default_once_is_omitted_but_recurrence_is_kept(self):
        args = {"time": "2026-11-11T07:00:00+07:00", "repeat": "once"}
        fixed, _ = self.normalize("Đặt báo thức", "alarm.create", args, [])
        self.assertNotIn("repeat", fixed)
        args["repeat"] = "daily"
        fixed, _ = self.normalize("Đặt báo thức mỗi ngày", "alarm.create", args, [])
        self.assertEqual(fixed["repeat"], "daily")

    def test_direct_clock_does_not_overwrite_arrival_departure_calculations(self):
        args = {"content": "rời nhà", "time": "2026-11-11T08:30:00+07:00"}
        fixed, _ = self.normalize("Chín giờ sáng mai tôi phải có mặt ở bệnh viện, nhắc tôi rời nhà sao cho đến sớm", "reminders.create", args, [])
        self.assertEqual(fixed, args)

    def test_correction_and_day_before_are_left_unchanged(self):
        args = {"time": "2026-11-11T05:30:00+07:00"}
        fixed, _ = self.normalize("Báo thức sáng mai năm giờ, à không, năm rưỡi thôi", "alarm.create", args, [])
        self.assertEqual(fixed, args)
        args = {"content": "chuẩn bị", "time": "2026-11-12T18:00:00+07:00"}
        fixed, _ = self.normalize("Thứ sáu hẹn lấy hàng, sáu giờ chiều hôm trước nhắc tôi chuẩn bị", "reminders.create", args, [])
        self.assertEqual(fixed, args)

    def test_implicit_half_day_is_preserved_and_spoken_minutes_are_parsed(self):
        args = {"content": "gọi lại", "time": "2026-11-10T17:00:00+07:00"}
        fixed, _ = self.normalize("Năm giờ nhắc tôi gọi lại", "reminders.create", args, [])
        self.assertEqual(fixed, args)

    def test_departure_uses_observed_route_duration_and_arrival_margin(self):
        steps = [{"kind": "call", "name": "maps.get_route", "arguments": {},
                  "result": {"route": {"duration_min": 20}}}]
        args = {"content": "rời nhà", "time": "2026-11-11T08:00:00+07:00"}
        fixed, _ = self.normalize("Tám giờ sáng mai phải có mặt, nhắc tôi rời nhà sao cho tới sớm mười phút", "reminders.create", args, steps)
        self.assertEqual(fixed["time"], "2026-11-11T07:30:00+07:00")

    def test_person_named_mai_is_not_tomorrow_and_night_clock_is_not_guessed(self):
        args = {"content": "gọi cô Mai", "time": "2026-11-10T20:00:00+07:00"}
        fixed, _ = self.normalize("Nhắc tôi gọi cô Mai lúc tám giờ tối nay", "reminders.create", args, [])
        self.assertEqual(fixed, args)

    def test_bare_person_mai_and_eleven_noon_do_not_shift_day_or_half_day(self):
        args = {"content": "gọi lại", "time": "2026-11-10T20:00:00+07:00"}
        fixed, _ = self.normalize("Nhắc Mai gọi lại lúc tám giờ tối nay", "reminders.create", args, [])
        self.assertEqual(fixed, args)
        args = {"content": "gọi lại", "time": "2026-11-10T11:00:00+07:00"}
        fixed, _ = self.normalize("Mười một giờ trưa nay nhắc tôi gọi lại", "reminders.create", args, [])
        self.assertEqual(fixed, args)

    def test_historical_calendar_update_advances_the_observed_anchor(self):
        self.row["history"] = [
            {"role": "assistant", "tool_calls": [{"name": "calendar.list_events", "arguments": {}}],
             "tool_results": [{"events": self.events}]},
            {"role": "assistant", "tool_calls": [{"name": "calendar.update_event", "arguments": {
                "event_id": "event_fixture", "start": "2026-11-10T17:00:00+07:00", "end": "2026-11-10T18:00:00+07:00"}}],
             "tool_results": [{"ok": True, "event": dict(self.events[0], start="2026-11-10T17:00:00+07:00", end="2026-11-10T18:00:00+07:00")}]},
        ]
        args = {"event_id": "event_fixture", "start": "2026-11-10T17:30:00+07:00", "end": "2026-11-10T18:30:00+07:00"}
        fixed, _ = self.normalize("Dời lịch khám răng muộn ba mươi phút", "calendar.update_event", args, [])
        self.assertEqual(fixed, args)

    def test_clock_date_belongs_to_its_clause_not_other_tasks(self):
        args = {"content": "uống thuốc", "time": "2026-11-11T08:00:00+07:00"}
        fixed, _ = self.normalize("Hôm nay tôi đi bộ rồi, mai tám giờ sáng nhắc tôi uống thuốc", "reminders.create", args, [])
        self.assertEqual(fixed, args)
        args = {"content": "nộp báo cáo", "time": "2026-11-13T08:00:00+07:00"}
        fixed, _ = self.normalize("Trưa nay đã dịch báo cáo, nhắc tôi tám giờ sáng thứ sáu nộp báo cáo", "reminders.create", args, [])
        self.assertEqual(fixed, args)

    def test_clock_for_booking_or_flight_filter_is_not_a_reminder_clock(self):
        args = {"time": "2026-11-11T16:00:00+07:00"}
        fixed, _ = self.normalize("Đặt vé sau mười hai giờ, báo thức trước ba tiếng", "alarm.create", args, [])
        self.assertEqual(fixed, args)
        args = {"content": "đủ chỗ", "time": "2026-11-10T17:00:00+07:00"}
        fixed, _ = self.normalize("Giữ chỗ từ hai giờ chiều nay, nhắc sau ba tiếng là đủ", "reminders.create", args, [])
        self.assertEqual(fixed, args)
        args = {"time": "2026-11-11T01:00:00+07:00"}
        fixed, _ = self.normalize("Đặt báo thức một giờ đêm mai", "alarm.create", args, [])
        self.assertEqual(fixed, args)
        args["time"] = "2026-11-10T09:50:00+07:00"
        fixed, _ = self.normalize("Chín giờ năm mươi nhắc tôi ra lấy đồ", "reminders.create", args, [])
        self.assertEqual(fixed, args)


if __name__ == "__main__":
    unittest.main()
