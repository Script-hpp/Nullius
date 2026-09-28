from repro_agent.setup_details import _chunks, _parse_list


def test_parse_takes_last_valid_list_after_thinking():
    content = 'Thinking [about 3 items] ... final:\n[{"topic": "lr", "value": "0.01", "quote": "q"}]'
    assert _parse_list(content) == [{"topic": "lr", "value": "0.01", "quote": "q"}]


def test_parse_no_list_gives_empty():
    assert _parse_list("no json here") == []


def test_chunks_cover_text_and_respect_size():
    text = "\n\n".join(f"paragraph {i} " + "x" * 300 for i in range(40))
    chunks = _chunks(text, size=2000)
    assert len(chunks) > 3
    assert all(len(c) < 2500 for c in chunks)
    assert "paragraph 39" in chunks[-1]
