#!/usr/bin/env python3
"""Storage 层 headless 测试 — 无需 PyQt5，可在任意环境运行。

用法:
    python3 test_storage.py
    MEETING_TRACKER_DB=./data/test_custom.db python3 test_storage.py
"""

import sys
import os

# 确保能找到 core/ 和 storage/
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 使用独立测试数据库
os.environ.setdefault("MEETING_TRACKER_DB",
                      os.path.join(_PROJECT_ROOT, "data", "test_meeting_tracker.db"))


def test_init():
    """测试: 数据库初始化 + 迁移。"""
    from storage.db import init_db, reset_db, db_path
    from storage.migration import run_migrations

    # 清理 + 重建
    reset_db()

    # 运行迁移 (应该是 0，因为 001 已通过 models.create_all 创建)
    applied = run_migrations()
    db_file = db_path()
    assert os.path.exists(db_file), f"DB file not created: {db_file}"
    print(f"  PASS: test_init (DB at {db_file}, {applied} migration(s) applied)")


def test_participant_crud():
    """测试: Participant CRUD + 唯一约束 + 批量导入。"""
    from storage.db import session_scope
    from storage.repo import ParticipantRepo

    with session_scope() as s:
        repo = ParticipantRepo(s)

        # Create
        p = repo.create("A001", "王强", organization="XX大学", role="项目负责人")
        assert p.tag_id == "A001"
        assert p.name == "王强"

        # Read by tag_id
        found = repo.get_by_tag_id("A001")
        assert found is not None
        assert found.organization == "XX大学"

        # Update
        repo.update(p, name="王强v2", title="队长")
        assert repo.get_by_id(p.id).name == "王强v2"
        assert repo.get_by_id(p.id).title == "队长"

        # List
        all_p = repo.list_all()
        assert len(all_p) == 1

        # Search
        results = repo.list_all(search="王强")
        assert len(results) == 1
        results = repo.list_all(search="NONEXIST")
        assert len(results) == 0

        # Unique constraint
        try:
            repo.create("A001", "重复的人")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected

        # Delete
        repo.delete(p)
        assert repo.get_by_tag_id("A001") is None

    print("  PASS: test_participant_crud")


def test_bulk_import():
    """测试: 批量导入人员。"""
    from storage.db import session_scope
    from storage.repo import ParticipantRepo

    with session_scope() as s:
        repo = ParticipantRepo(s)

        result = repo.bulk_import([
            {"tag_id": "A001", "name": "王强", "role": "项目负责人"},
            {"tag_id": "A002", "name": "李老师", "role": "评委"},
            {"tag_id": "A003", "name": "", "role": "主持人"},       # 应报错 (name 为空)
            {"tag_id": "", "name": "无名氏"},                       # 应报错 (tag_id 为空)
        ])

        assert result["created"] == 2
        assert result["updated"] == 0
        assert len(result["errors"]) == 2

        # Re-import: 应更新已有记录
        result2 = repo.bulk_import([
            {"tag_id": "A001", "name": "王强(更新)", "role": "队长"},
        ])
        assert result2["updated"] == 1
        assert repo.get_by_tag_id("A001").name == "王强(更新)"

        # Cleanup
        for p in repo.list_all():
            repo.delete(p)

    print("  PASS: test_bulk_import")


def test_meeting_lifecycle():
    """测试: Meeting 生命周期 + 状态迁移。"""
    from storage.db import session_scope
    from storage.repo import MeetingRepo

    with session_scope() as s:
        repo = MeetingRepo(s)

        # Create
        m = repo.create("项目路演", location="会议室A")
        assert m.status == "planned"
        assert m.start_time is None

        # Invalid transition: planned → completed
        try:
            repo.end_meeting(m)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected

        # Start
        repo.start_meeting(m)
        assert m.status == "in_progress"
        assert m.start_time is not None

        # Only one active
        assert repo.get_active() is not None

        # End
        repo.end_meeting(m)
        assert m.status == "completed"
        assert m.end_time is not None

        # Active should be None now
        assert repo.get_active() is None

        # Cancel another meeting
        m2 = repo.create("测试会议2")
        repo.cancel_meeting(m2)
        assert m2.status == "cancelled"

    print("  PASS: test_meeting_lifecycle")


def test_speaker_segments():
    """测试: SpeakerSegment 开始/结束/活跃片段。"""
    from storage.db import session_scope
    from storage.repo import MeetingRepo, SpeakerSegmentRepo

    with session_scope() as s:
        mr = MeetingRepo(s)
        sr = SpeakerSegmentRepo(s)

        m = mr.create("测试会议")
        mr.start_meeting(m)

        # Start first segment
        seg1 = sr.start_segment(m.id, "A001", "王强", "AprilTag")
        assert seg1.end_time is None  # Active
        assert sr.get_active_segment(m.id) is not None

        # Start second segment — should auto-end seg1
        seg2 = sr.start_segment(m.id, "A002", "李老师", "AprilTag")
        assert seg2.end_time is None

        # Reload seg1 — should now have end_time
        from storage.models import SpeakerSegment
        seg1_reloaded = s.get(SpeakerSegment, seg1.id)
        assert seg1_reloaded.end_time is not None
        assert seg1_reloaded.duration_seconds >= 0

        # End seg2
        sr.end_segment(seg2)
        assert seg2.end_time is not None
        assert seg2.duration_seconds >= 0
        assert sr.get_active_segment(m.id) is None

        # Duration stats
        total = sr.get_total_duration(m.id)
        assert total > 0

        # By speaker
        t_wang = sr.get_total_duration(m.id, speaker_tag_id="A001")
        assert t_wang >= 0

    print("  PASS: test_speaker_segments")


def test_event_logging():
    """测试: Event 单条 + 批量插入 + 查询。"""
    from storage.db import session_scope
    from storage.repo import EventRepo
    from datetime import datetime

    with session_scope() as s:
        repo = EventRepo(s)

        # Single event
        repo.log_event("meeting_started", meeting_id=1,
                       payload={"name": "test", "location": "A"})

        # System-wide event (no meeting)
        repo.log_event("system_warning", payload={"msg": "CPU usage high"})

        # Batch insert
        repo.log_batch([
            {"event_type": "tag_detected", "meeting_id": 1,
             "payload": {"tag_id": "A001"}},
            {"event_type": "tag_detected", "meeting_id": 1,
             "payload": {"tag_id": "A002"}},
            {"event_type": "servo_moved", "meeting_id": 1,
             "payload": {"h_angle": 45.0}},
        ])

        # Query recent
        recent = repo.get_recent(minutes=60)
        assert len(recent) == 5, f"Expected 5 events, got {len(recent)}"

        # Query by meeting
        meeting_events = repo.get_for_meeting(1)
        assert len(meeting_events) == 4  # all except system_warning

        # Query by type
        tag_events = repo.get_for_meeting(1, event_types=["tag_detected"])
        assert len(tag_events) == 2

        # Count
        assert repo.count(event_type="tag_detected") == 2
        assert repo.count(event_type="system_warning") == 1

        # System-wide event has meeting_id=None
        sw = [e for e in recent if e.event_type == "system_warning"]
        assert len(sw) == 1
        assert sw[0].meeting_id is None

    print("  PASS: test_event_logging")


def test_host_notes():
    """测试: HostNote CRUD。"""
    from storage.db import session_scope
    from storage.repo import HostNoteRepo

    with session_scope() as s:
        repo = HostNoteRepo(s)

        n1 = repo.create_note(
            meeting_id=1,
            note_type="评委问题",
            content="李老师询问系统精度指标",
            related_speaker="李老师",
        )

        n2 = repo.create_note(
            meeting_id=1,
            note_type="待办事项",
            content="补充手机提醒功能测试",
        )

        # Query
        all_notes = repo.get_notes_for_meeting(1)
        assert len(all_notes) == 2

        # Query by type
        judge_notes = repo.get_notes_for_meeting(1, note_type="评委问题")
        assert len(judge_notes) == 1
        assert judge_notes[0].related_speaker == "李老师"

    print("  PASS: test_host_notes")


def test_config():
    """测试: SystemConfig 存取 + 类型转换。"""
    from storage.db import session_scope
    from storage.repo import ConfigRepo

    with session_scope() as s:
        repo = ConfigRepo(s)

        # Set/get scalars
        repo.set("fusion", "gain_h", 0.7)
        assert repo.get("fusion", "gain_h") == 0.7

        repo.set("fusion", "threshold_audio", 10.0)
        assert repo.get("fusion", "threshold_audio") == 10.0

        repo.set("vad", "enabled", True)
        assert repo.get("vad", "enabled") is True

        repo.set("vad", "device", "hw:1,0")
        assert repo.get("vad", "device") == "hw:1,0"

        # Default
        assert repo.get("nonexistent", "key", default=42) == 42

        # Section bulk
        repo.set_section("test_section", {"a": 1, "b": "hello", "c": [1, 2, 3]})
        section = repo.get_section("test_section")
        assert section == {"a": 1, "b": "hello", "c": [1, 2, 3]}

        # Delete section
        repo.delete_section("test_section")
        assert repo.get_section("test_section") == {}

    print("  PASS: test_config")


def test_concurrent_sessions():
    """测试: 多线程并发写入 (验证 scoped_session 线程安全)。"""
    import threading

    from storage.db import session_scope, get_session_factory
    from storage.repo import EventRepo

    errors = []
    results = []

    def worker(thread_id: int):
        try:
            with session_scope() as session:
                repo = EventRepo(session)
                for i in range(20):
                    repo.log_event(
                        "test_concurrent",
                        payload={"thread": thread_id, "seq": i}
                    )
            results.append(thread_id)
        except Exception as e:
            errors.append((thread_id, str(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent errors: {errors}"
    assert len(results) == 5

    # Verify total count
    with session_scope() as session:
        repo = EventRepo(session)
        count = repo.count(event_type="test_concurrent")
        assert count == 100, f"Expected 100 events, got {count}"

    print(f"  PASS: test_concurrent_sessions (100 events from 5 threads)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running storage layer tests...\n")

    # 初始化
    test_init()

    # CRUD
    test_participant_crud()
    test_bulk_import()
    test_meeting_lifecycle()

    # 发言片段
    test_speaker_segments()

    # 事件和备注
    test_event_logging()
    test_host_notes()

    # 配置
    test_config()

    # 并发安全
    test_concurrent_sessions()

    print(f"\n{'='*50}")
    print("All tests passed!")
    print(f"{'='*50}")
