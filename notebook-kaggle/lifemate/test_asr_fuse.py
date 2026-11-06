"""CPU checks for the word-level ASR fusion (lm-A-054)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "agent-qwen35-fuse"))
from asr_fuse import SpanModel, TrigramLM, fuse, number_words, span_features, spoken, spoken_text

LM = TrigramLM(["trả tiền mạng giúp chị", "trả tiền mạng bằng ví", "nhắc tôi uống thuốc lúc tám giờ", "chuyển một trăm hai mươi nghìn"] * 5)


class AsrFuseTests(unittest.TestCase):
    def test_numbers_read_aloud(self):
        self.assertEqual(number_words("15"), ["mười", "lăm"])
        self.assertEqual(number_words("45"), "bốn mươi lăm".split())
        self.assertEqual(number_words("118"), "một trăm mười tám".split())
        self.assertEqual(number_words("105"), "một trăm linh năm".split())
        self.assertEqual(number_words("100000"), "một trăm nghìn".split())
        self.assertEqual(number_words("1500000"), "một triệu năm trăm nghìn".split())
        self.assertEqual(number_words("0912"), "không chín một hai".split())
        self.assertEqual(spoken("100.000đ,"), ["100000đ"])
        self.assertEqual(spoken("100.000,"), "một trăm nghìn".split())

    def test_same_words_keep_primary_format(self):
        self.assertEqual(fuse("Nhắc tôi lúc 8 giờ.", "NHẮC TÔI LÚC TÁM GIỜ", LM, set()), "Nhắc tôi lúc 8 giờ.")

    def test_language_model_repairs_a_word(self):
        self.assertEqual(fuse("Trả tiền mác giúp chị.", "TRẢ TIỀN MẠNG GIÚP CHỊ", LM, set()), "Trả tiền mạng giúp chị.")

    def test_context_word_breaks_the_tie(self):
        with_context = fuse("Gọi Tấn nhé.", "GỌI TUẤN NHÉ", LM, {"tuấn"}, bias=0, context_weight=4)
        self.assertEqual(with_context, "Gọi Tuấn nhé.")

    def test_primary_kept_when_secondary_is_unlikely(self):
        self.assertEqual(fuse("Trả tiền mạng giúp chị.", "TRẢ TIỀN MÁC GIÚP CHỊ", LM, set()), "Trả tiền mạng giúp chị.")

    def test_span_inside_a_number_loses_nothing(self):
        for other in ("HAI TRĂM MƯỜI TÁM NGHÌN", "MỘT TRĂM HAI MƯƠI TÁM NGHÌN", "MỘT TRĂM MƯỜI BA NGHÌN"):
            self.assertEqual(fuse("Chuyển 118 nghìn nhé", "CHUYỂN " + other + " NHÉ", LM, set()), "Chuyển 118 nghìn nhé", other)

    def test_every_word_survives_in_some_form(self):
        out = fuse("Trả tiền mác giúp chị bằng ví", "TRẢ TIỀN MẠNG GIÚP", LM, set(), bias=100)
        self.assertEqual(out, "Trả tiền mác giúp chị bằng ví")

    def test_empty_inputs_return_primary(self):
        self.assertEqual(fuse("", "CÓ CHỮ", LM, set()), "")
        self.assertEqual(fuse("Có chữ", "", LM, set()), "Có chữ")

    def test_span_model_decides_and_keeps_format(self):
        take = SpanModel([0.0] * 9 + [1.0], [0.0] * 10, [1.0] * 10)
        keep = SpanModel([0.0] * 9 + [-1.0], [0.0] * 10, [1.0] * 10)
        self.assertEqual(fuse("Trả tiền mác lúc 8 giờ.", "TRẢ TIỀN MẠNG LÚC TÁM GIỜ", LM, set(), model=take), "Trả tiền mạng lúc 8 giờ.")
        self.assertEqual(fuse("Trả tiền mác lúc 8 giờ.", "TRẢ TIỀN MẠNG LÚC TÁM GIỜ", LM, set(), model=keep), "Trả tiền mác lúc 8 giờ.")

    def test_span_model_cannot_cut_a_number(self):
        take = SpanModel([0.0] * 9 + [1.0], [0.0] * 10, [1.0] * 10)
        self.assertEqual(fuse("Chuyển 118 nghìn nhé", "CHUYỂN HAI TRĂM MƯỜI TÁM NGHÌN NHÉ", LM, set(), model=take), "Chuyển 118 nghìn nhé")

    def test_span_features_shape(self):
        f = span_features(LM, ["<s>", "<s>"], ["mác"], ["mạng"], ["giúp", "chị"], {"mạng"}, {"tiền"})
        self.assertEqual(len(f), 10)
        self.assertEqual(f[-1], 1.0)
        self.assertEqual(f[3], -1)

    def test_spoken_text_is_stable(self):
        self.assertEqual(spoken_text("Mai 9 giờ, nhé!"), ["mai", "chín", "giờ", "nhé"])


if __name__ == "__main__":
    unittest.main()
