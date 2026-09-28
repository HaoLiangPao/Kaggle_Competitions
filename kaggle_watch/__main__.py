"""Discover new Kaggle competitions without downloading or entering them."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .profiles import PROFILES


DEFAULT_STATE = Path(__file__).resolve().parent.parent / ".local" / "kaggle_watch.sqlite3"


@dataclass(frozen=True)
class Competition:
    slug: str
    url: str
    deadline: date
    category: str
    reward: str
    team_count: int


def parse_competitions(output: str, today: date) -> list[Competition]:
    """Parse Kaggle CSV, tolerating CLI notices before its header."""
    lines = output.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith("ref,deadline,")), None)
    if header is None:
        raise ValueError("Kaggle CLI did not return a competition CSV header")

    results: dict[str, Competition] = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[header:]))):
        url = (row.get("ref") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.netloc not in {"www.kaggle.com", "kaggle.com"}:
            continue
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "competitions" or not parts[1]:
            continue
        try:
            deadline = datetime.fromisoformat(row["deadline"].strip()).date()
            team_count = int(row.get("teamCount") or 0)
        except (TypeError, ValueError, KeyError):
            continue
        if deadline < today:
            continue
        competition = Competition(
            slug=parts[1],
            url=url,
            deadline=deadline,
            category=(row.get("category") or "Unknown").strip(),
            reward=(row.get("reward") or "").strip(),
            team_count=team_count,
        )
        results[competition.slug] = competition
    return list(results.values())


def fetch_competitions(pages: int, token_file: Path | None, today: date, sort_by: str = "recentlyCreated") -> list[Competition]:
    env = os.environ.copy()
    if token_file is not None:
        token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Token file is empty")
        env["KAGGLE_API_TOKEN"] = token

    found: dict[str, Competition] = {}
    for page in range(1, pages + 1):
        result = subprocess.run(
            ["kaggle", "competitions", "list", "--group", "general", "--sort-by", sort_by, "-v", "-p", str(page)],
            capture_output=True,
            text=True,
            timeout=45,
            env=env,
            check=False,
        )
        if result.returncode:
            # Kaggle diagnostics may contain account details; keep errors concise.
            raise RuntimeError("Kaggle CLI list failed; check CLI authentication and network access")
        if "No competitions found" in result.stdout:
            break
        page_competitions = parse_competitions(result.stdout, today)
        if sort_by == "latestDeadline" and not page_competitions:
            break
        found.update({item.slug: item for item in page_competitions})
    return list(found.values())


def open_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS competitions (
            slug TEXT PRIMARY KEY,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('baseline', 'pending', 'sent')),
            url TEXT,
            deadline TEXT,
            category TEXT,
            reward TEXT,
            team_count INTEGER
        )"""
    )
    # Upgrade the first experimental state database without losing its baseline.
    columns = {row[1] for row in connection.execute("PRAGMA table_info(competitions)")}
    for name, kind in (
        ("url", "TEXT"),
        ("deadline", "TEXT"),
        ("category", "TEXT"),
        ("reward", "TEXT"),
        ("team_count", "INTEGER"),
    ):
        if name not in columns:
            connection.execute(f"ALTER TABLE competitions ADD COLUMN {name} {kind}")
    connection.commit()
    return connection


def update_state(connection: sqlite3.Connection, competitions: list[Competition], now: str) -> tuple[bool, list[Competition]]:
    """First scan establishes a baseline; pending cards survive pagination changes."""
    initial = connection.execute("SELECT COUNT(*) FROM competitions").fetchone()[0] == 0
    existing = set(row[0] for row in connection.execute("SELECT slug FROM competitions"))
    with connection:
        for item in competitions:
            values = (
                item.url, item.deadline.isoformat(), item.category,
                item.reward, item.team_count,
            )
            if item.slug in existing:
                connection.execute(
                    """UPDATE competitions
                       SET last_seen = ?, url = ?, deadline = ?, category = ?,
                           reward = ?, team_count = ?
                       WHERE slug = ?""",
                    (now, *values, item.slug),
                )
            else:
                status = "baseline" if initial else "pending"
                connection.execute(
                    """INSERT INTO competitions
                       (slug, first_seen, last_seen, status, url, deadline, category, reward, team_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (item.slug, now, now, status, *values),
                )
                existing.add(item.slug)
    today = datetime.fromisoformat(now).date().isoformat()
    rows = connection.execute(
        """SELECT slug, url, deadline, category, reward, team_count
           FROM competitions
           WHERE status = 'pending' AND deadline >= ?
           ORDER BY first_seen, slug""",
        (today,),
    )
    pending = [
        Competition(slug, url, date.fromisoformat(deadline), category, reward, team_count)
        for slug, url, deadline, category, reward, team_count in rows
    ]
    return initial, pending


def mark_sent(connection: sqlite3.Connection, slugs: list[str]) -> None:
    """Acknowledge only pending competitions after confirmed delivery."""
    if not slugs:
        raise ValueError("At least one --slug is required for ack")
    unique_slugs = list(dict.fromkeys(slugs))
    placeholders = ",".join("?" for _ in unique_slugs)
    rows = connection.execute(
        f"SELECT slug, status FROM competitions WHERE slug IN ({placeholders})",
        unique_slugs,
    ).fetchall()
    statuses = dict(rows)
    if any(statuses.get(slug) != "pending" for slug in unique_slugs):
        raise ValueError("All acknowledged slugs must exist and be pending")
    with connection:
        connection.executemany(
            "UPDATE competitions SET status = 'sent' WHERE slug = ?",
            [(slug,) for slug in unique_slugs],
        )


def card(item: Competition) -> str:
    learning_hint = (
        "适合先看：入门或 Playground 类别；具体任务仍需查看比赛页面。"
        if item.category.lower() in {"playground", "gettingstarted"}
        else "先评估任务、数据规模和规则，再决定是否参加。"
    )
    return (
        f"• <{item.url}|{item.slug}>\n"
        f"  类别：{item.category}｜截止：{item.deadline.isoformat()}｜队伍：{item.team_count}｜奖励：{item.reward or '未列出'}\n"
        f"  {learning_hint}"
    )


def message(competitions: list[Competition], title: str = "新发现的 Kaggle 比赛") -> str:
    return f"*{title}*\n" + "\n".join(card(item) for item in competitions) + (
        "\n_本次扫描仅依据 Kaggle 比赛列表；不会下载或分析数据，也不会加入比赛。_"
    )


def recommendation(item: Competition, today: date) -> tuple[int, int, str]:
    """Return subjective learning difficulty and current fit, both 1–5."""
    profile = PROFILES[item.slug]
    remaining = (item.deadline - today).days
    urgency_penalty = 2 if remaining <= 7 else 1 if remaining <= 14 else 0
    return profile["difficulty"], max(1, profile["fit"] - urgency_penalty), profile["reason"]


def report_messages(competitions: list[Competition], today: date) -> tuple[str, str]:
    """Create a mobile-friendly summary and a complete list for a Slack thread."""
    timed = sorted(
        (item for item in competitions if item.category.lower() != "getting started"),
        key=lambda item: (item.deadline, item.slug),
    )
    practice = sorted(
        (item for item in competitions if item.category.lower() == "getting started"),
        key=lambda item: item.slug,
    )
    profiled = [item for item in competitions if item.slug in PROFILES]
    profiled.sort(key=lambda item: (-recommendation(item, today)[1], recommendation(item, today)[0]))
    top = profiled[:3]
    lines = [
        f"*Kaggle 比赛雷达｜{today.isoformat()}*",
        f"Kaggle CLI general 列表：*{len(competitions)} 场未截止*；限期赛事 {len(timed)}，入门练习 {len(practice)}。",
        "这是在赛快照，不代表这些比赛都是新发布。",
        "",
        "*推荐从这里开始*（难度：1 易 → 5 难；适合度：1 低 → 5 高）",
    ]
    for rank, item in enumerate(top, 1):
        profile = PROFILES[item.slug]
        difficulty, fit, reason = recommendation(item, today)
        size_mb = profile["train_test_bytes"] / 1_000_000
        deadline = "长期练习" if item.category.lower() == "getting started" else item.deadline.isoformat()
        lines.extend([
            f"{rank}. <{item.url}|{profile['title']}>｜难度 {difficulty}/5｜适合 {fit}/5",
            f"   {profile['task']}；评分：{profile['metric']}；train+test 约 {size_mb:.2f} MB；{deadline}",
            f"   推荐理由：{reason}",
        ])
    lines.extend([
        "",
        "评分是学习路线估计，考虑任务复杂度、数据体量、现有代码与剩余时间；仅对 2026-09-27 已核对规则的比赛给分。",
        f"完整 {len(competitions)} 场清单见本消息的回复。数据体量来自 Kaggle 文件列表；生成本报告时未下载新数据。",
    ])
    catalog = ["*限期赛事*"]
    for item in timed:
        catalog.append(
            f"• {item.deadline.isoformat()}｜{item.category}｜<{item.url}|{item.slug}>"
        )
    catalog.extend(["", "*入门练习（CLI 列表显示 2030 年截止）*"])
    for item in practice:
        catalog.append(f"• <{item.url}|{item.slug}>")
    return "\n".join(lines), "\n".join(catalog)


def send_slack(text: str) -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        raise ValueError("SLACK_WEBHOOK_URL is required for --send-slack")
    parsed = urlparse(webhook)
    if parsed.scheme != "https" or parsed.netloc != "hooks.slack.com":
        raise ValueError("SLACK_WEBHOOK_URL must be an HTTPS hooks.slack.com URL")
    request = Request(
        webhook,
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            if response.status != 200 or response.read().strip() != b"ok":
                raise RuntimeError("Slack webhook did not confirm delivery")
    except Exception:
        # HTTP errors can include the secret URL; never print the underlying exception.
        raise RuntimeError("Slack webhook did not confirm delivery") from None


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview or track newly listed Kaggle competitions")
    parser.add_argument("action", choices=("preview", "scan", "ack", "report"))
    parser.add_argument("--token-file", type=Path, help="Ignored local file containing a Kaggle API token")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--pages", type=int, help="Listing pages: default 3 for scan, 10 for report")
    parser.add_argument("--limit", type=int, default=5, help="Maximum cards to display or send (default: 5)")
    parser.add_argument("--timezone", default="America/Toronto")
    parser.add_argument("--send-slack", action="store_true", help="Send pending cards via SLACK_WEBHOOK_URL")
    parser.add_argument("--slug", action="append", default=[], help="Pending competition slug to acknowledge after connector delivery")
    parser.add_argument("--json", action="store_true", help="Print report as JSON for an agent or Slack connector")
    args = parser.parse_args()
    if (args.pages is not None and args.pages < 1) or args.limit < 1:
        parser.error("--pages and --limit must be positive")
    if args.send_slack and args.action != "scan":
        parser.error("--send-slack requires scan")

    try:
        if args.action == "ack":
            with closing(open_state(args.state)) as connection:
                mark_sent(connection, args.slug)
            print(f"已确认发送 {len(set(args.slug))} 场比赛。")
            return 0
        now = datetime.now(ZoneInfo(args.timezone))
        pages = args.pages or (10 if args.action == "report" else 3)
        sort_by = "latestDeadline" if args.action == "report" else "recentlyCreated"
        competitions = fetch_competitions(pages, args.token_file, now.date(), sort_by)
        if args.action == "report":
            summary, catalog = report_messages(competitions, now.date())
            if args.json:
                print(json.dumps({
                    "date": now.date().isoformat(),
                    "count": len(competitions),
                    "summary": summary,
                    "catalog": catalog,
                }, ensure_ascii=False))
            else:
                print(summary + "\n\n--- 完整清单 ---\n" + catalog)
            return 0
        if args.action == "preview":
            print(message(competitions[: args.limit], title="Kaggle 比赛预览") if competitions else "近期列表中没有未截止比赛。")
            return 0

        with closing(open_state(args.state)) as connection:
            initial, pending = update_state(connection, competitions, now.isoformat())
            if initial:
                print(f"已建立基线：记录 {len(competitions)} 场未截止比赛；本次不发送通知。")
                return 0
            if not pending:
                print("没有待通知的新比赛。")
                return 0
            selected = pending[: args.limit]
            text = message(selected)
            print(text)
            if args.send_slack:
                send_slack(text)
                with connection:
                    connection.executemany(
                        "UPDATE competitions SET status = 'sent' WHERE slug = ?",
                        [(item.slug,) for item in selected],
                    )
                print(f"已发送 {len(selected)} 场比赛到 Slack。")
            else:
                print("预览模式：以上比赛保持待通知；添加 --send-slack 后发送。")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
