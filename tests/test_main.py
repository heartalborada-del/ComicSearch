import unittest

from app.main import parse_keyword_ids


class ParseKeywordIdsTests(unittest.TestCase):
    def test_parse_json_array(self):
        self.assertEqual(parse_keyword_ids("[1,2,3]"), [1, 2, 3])

    def test_parse_comma_separated(self):
        self.assertEqual(parse_keyword_ids("1, 2, 3"), [1, 2, 3])

    def test_empty(self):
        self.assertEqual(parse_keyword_ids(None), [])
        self.assertEqual(parse_keyword_ids(""), [])


if __name__ == "__main__":
    unittest.main()
