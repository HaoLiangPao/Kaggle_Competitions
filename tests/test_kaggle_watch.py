import tempfile
import unittest
from datetime import date
from pathlib import Path

from kaggle_watch.__main__ import Competition, mark_sent, message, open_state, parse_competitions, recommendation, report_messages, update_state


class KaggleWatchTests(unittest.TestCase):
    def test_parse_filters_expired_and_deduplicates(self):
        output = (
            "CLI update notice\n"
            "ref,deadline,category,reward,teamCount,userHasEntered\n"
            "https://www.kaggle.com/competitions/new-one,2026-10-01 23:59:00,Playground,Swag,31,False\n"
            "https://www.kaggle.com/competitions/old-one,2026-09-01 23:59:00,Featured,Prize,41,False\n"
            "https://www.kaggle.com/competitions/new-one,2026-10-01 23:59:00,Playground,Swag,31,False\n"
        )
        found = parse_competitions(output, date(2026, 9, 27))
        self.assertEqual([item.slug for item in found], ["new-one"])

    def test_baseline_then_pending_then_no_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            first = Competition(
                "first", "https://www.kaggle.com/competitions/first",
                date(2026, 10, 1), "Playground", "Swag", 20
            )
            second = Competition(
                "second", "https://www.kaggle.com/competitions/second",
                date(2026, 10, 2), "Featured", "", 30
            )
            connection = open_state(path)
            self.addCleanup(connection.close)
            initial, pending = update_state(connection, [first], "2026-09-27T10:00:00")
            self.assertTrue(initial)
            self.assertEqual(pending, [])
            initial, pending = update_state(connection, [first, second], "2026-09-28T10:00:00")
            self.assertFalse(initial)
            self.assertEqual([item.slug for item in pending], ["second"])
            # A pending item remains deliverable when it leaves the recent listing pages.
            _, pending = update_state(connection, [first], "2026-09-28T11:00:00")
            self.assertEqual([item.slug for item in pending], ["second"])
            with self.assertRaises(ValueError):
                mark_sent(connection, ["first"])
            mark_sent(connection, ["second"])
            _, pending = update_state(connection, [first, second], "2026-09-29T10:00:00")
            self.assertEqual(pending, [])
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM competitions").fetchone()[0], 2)

    def test_report_scores_and_lists_all_competitions(self):
        today = date(2026, 9, 27)
        titanic = Competition(
            "titanic", "https://www.kaggle.com/competitions/titanic",
            date(2030, 1, 1), "Getting Started", "Knowledge", 100
        )
        playground = Competition(
            "playground-series-s6e9",
            "https://www.kaggle.com/competitions/playground-series-s6e9",
            date(2026, 9, 30), "Playground", "Swag", 200
        )
        summary, catalog = report_messages([titanic, playground], today)
        self.assertIn("2 场未截止", summary)
        self.assertIn("完整 2 场清单", summary)
        self.assertIn("Titanic 生存预测", summary)
        self.assertIn("难度 1/5｜适合 5/5", summary)
        self.assertEqual(recommendation(playground, today)[1], 2)
        self.assertIn("playground-series-s6e9", catalog)
        self.assertIn("titanic", catalog)

    def test_message_does_not_claim_dataset_analysis(self):
        competition = Competition(
            "first", "https://www.kaggle.com/competitions/first",
            date(2026, 10, 1), "Playground", "Swag", 20
        )
        text = message([competition])
        self.assertIn("不会下载或分析数据", text)
        self.assertIn("first", text)


if __name__ == "__main__":
    unittest.main()
