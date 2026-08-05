"""音频服务的字节范围（Range）行为。

为什么单独测这个：线上朗读坏过一次，症状是按钮永远停在「生成语音…」，
既没有报错也没有失败的网络请求。原因是响应里挂了 Accept-Ranges: bytes
却从来不处理 Range 请求——Chrome 的 <audio> 看到「可字节定位」就先发
Range: bytes=0-，收到的却是普通 200 整段响应，于是卡在 readyState=0。

这类缺陷不会让任何接口返回错误码，只会让播放器安静地不动，
所以只能在这一层直接对着字节断言。
"""

import sys
from http import HTTPStatus
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class _Recorder:
    """替身处理器：只记录状态行、响应头和写出的字节。"""

    def __init__(self, range_header=None):
        self.headers = {"Range": range_header} if range_header else {}
        self.status = None
        self.sent = {}
        self.body = b""
        self.close_connection = False

    # 被测方法用到的那几个 BaseHTTPRequestHandler 接口
    def _headers(self, status, content_type, length, extra=None):
        self.status = status
        self.sent = dict(extra or {})
        self.sent["Content-Type"] = content_type
        if length is not None:
            self.sent["Content-Length"] = str(length)

    @property
    def wfile(self):
        recorder = self

        class _W:
            @staticmethod
            def write(data):
                recorder.body += data

            @staticmethod
            def flush():
                pass

        return _W()


def _handler(range_header=None):
    from web.server import ChatHandler
    rec = _Recorder(range_header)
    # headers 是个 dict，get 的语义与 email.Message 一致，够用
    rec.headers = _Headers(rec.headers)
    rec._send_audio_file = ChatHandler._send_audio_file.__get__(rec, _Recorder)
    return rec


class _Headers(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "a.mp3"
    path.write_bytes(bytes(range(256)) * 8)      # 2048 字节，内容可预测
    return path


def test_full_request_returns_whole_file(audio_file):
    h = _handler()
    h._send_audio_file(audio_file)
    assert h.status == HTTPStatus.OK
    assert h.body == audio_file.read_bytes()
    assert h.sent["Content-Length"] == "2048"


def test_accept_ranges_is_only_advertised_when_honored(audio_file):
    """宣称支持就必须真的支持。这两件事分开的时候播放器会静默卡死。"""
    h = _handler()
    h._send_audio_file(audio_file)
    assert h.sent.get("Accept-Ranges") == "bytes"

    served = _handler("bytes=0-")
    served._send_audio_file(audio_file)
    assert served.status == HTTPStatus.PARTIAL_CONTENT


def test_open_ended_range_is_partial_content(audio_file):
    """Chrome 的 <audio> 发的就是这一个：bytes=0-"""
    h = _handler("bytes=0-")
    h._send_audio_file(audio_file)
    assert h.status == HTTPStatus.PARTIAL_CONTENT
    assert h.sent["Content-Range"] == "bytes 0-2047/2048"
    assert h.body == audio_file.read_bytes()


def test_mid_file_range_returns_exact_slice(audio_file):
    """拖动进度条时发的形状。切错一个字节，播放器就会解出杂音。"""
    h = _handler("bytes=100-199")
    h._send_audio_file(audio_file)
    assert h.status == HTTPStatus.PARTIAL_CONTENT
    assert h.sent["Content-Range"] == "bytes 100-199/2048"
    assert h.sent["Content-Length"] == "100"
    assert h.body == audio_file.read_bytes()[100:200]


def test_suffix_range_takes_tail(audio_file):
    h = _handler("bytes=-50")
    h._send_audio_file(audio_file)
    assert h.status == HTTPStatus.PARTIAL_CONTENT
    assert h.body == audio_file.read_bytes()[-50:]


def test_range_past_end_is_rejected_not_truncated(audio_file):
    """越界要回 416。悄悄返回空体会被当成损坏的音频。"""
    h = _handler("bytes=5000-6000")
    h._send_audio_file(audio_file)
    assert h.status == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE
    assert h.sent["Content-Range"] == "bytes */2048"
    assert h.body == b""


def test_end_beyond_eof_clamps(audio_file):
    h = _handler("bytes=2000-9999")
    h._send_audio_file(audio_file)
    assert h.status == HTTPStatus.PARTIAL_CONTENT
    assert h.sent["Content-Range"] == "bytes 2000-2047/2048"
    assert h.body == audio_file.read_bytes()[2000:]


def test_malformed_range_falls_back_to_whole_file(audio_file):
    """看不懂的 Range 按整段发，别把它当成错误——播放总比不播好。"""
    for bad in ("bytes=abc", "items=0-10", "bytes=", "bytes=1-2, 5-6"):
        h = _handler(bad)
        h._send_audio_file(audio_file)
        assert h.status == HTTPStatus.OK, bad
        assert h.body == audio_file.read_bytes(), bad
