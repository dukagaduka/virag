"""utils: the periodic loop, the text fields of a streamed multipart body."""

from __future__ import annotations

import asyncio

import pytest

from virag.utils import MultipartFields, cancel, every


def test_every_runs_the_job_on_the_interval_and_survives_its_failures(caplog):
    runs = []

    async def job():
        runs.append(len(runs))

        if len(runs) == 2:
            raise RuntimeError("one bad run")

    async def go():
        task = asyncio.create_task(every(0.005, job, name="the job"))
        await asyncio.sleep(0.05)
        await cancel(task)

    asyncio.run(go())

    assert len(runs) >= 3                                               # went on after the failure
    assert "the job failed" in caplog.text and "one bad run" in caplog.text


def test_every_with_no_interval_never_runs():
    runs = []

    async def job():
        runs.append(1)

    asyncio.run(asyncio.wait_for(every(0, job, name="the job"), timeout=1))

    assert runs == []


# ---------------------------------------------------------------------------
# MultipartFields
# ---------------------------------------------------------------------------

_BODY = (
    b'--b\r\nContent-Disposition: form-data; name="doc_type"\r\n\r\nPDF\r\n'
    b'--b\r\nContent-Disposition: form-data; name="document"; filename="a.pdf"\r\n'
    b'Content-Type: application/pdf\r\n\r\n%PDF-' + b"x" * 5000 + b'\r\n'
    b'--b\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n{"name": "\xd0\x90"}\r\n'
    b'--b--\r\n'
)


@pytest.mark.parametrize("size", [1, 7, 64, len(_BODY)])
def test_text_fields_are_picked_out_however_the_body_is_cut(size):
    form = MultipartFields("multipart/form-data; boundary=b")

    for start in range(0, len(_BODY), size):
        form.feed(_BODY[start:start + size])

    assert form.fields == {"doc_type": "PDF", "metadata": '{"name": "А"}'}   # the file is passed over


def test_a_field_is_there_as_soon_as_it_has_gone_by():
    form = MultipartFields("multipart/form-data; boundary=b")
    head = _BODY[:_BODY.index(b"%PDF-")]

    form.feed(head)

    assert form.fields == {"doc_type": "PDF"}                           # the file is still streaming


def test_what_is_not_a_multipart_form_is_refused():
    for content_type in ("application/json", "multipart/form-data", ""):
        with pytest.raises(ValueError):
            MultipartFields(content_type)

    with pytest.raises(ValueError):
        MultipartFields("multipart/form-data; boundary=b").feed(b"--b\r\nnot a header\r\n\r\n")


def test_an_oversized_text_field_is_refused():
    form = MultipartFields("multipart/form-data; boundary=b")

    with pytest.raises(ValueError):
        form.feed(b'--b\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n' + b"x" * (form.MAX_FIELD_BYTES + 1))
