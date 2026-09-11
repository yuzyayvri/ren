import pytest

from scripts.phase5_client import SynthesisClient, SynthesisError, build_request

DECODING = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 7, "n_predict": 64}
PACKET = {"schema": "phase5-packet-v1", "case_id": "c1", "source_stage": "phase3-txl",
          "findings": [], "context": [], "limitations": []}


def _client(transport):
    return SynthesisClient("http://127.0.0.1:8080", "m", DECODING, "sys", transport=transport)


def test_request_shape_is_frozen():
    payload = build_request("sys", PACKET, DECODING, "m")
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0.0 and payload["seed"] == 7


def test_success_path_returns_parsed_json():
    body = {"choices": [{"message": {"content": '{"schema": "phase5-response-v1"}'}}]}
    _, parsed = _client(lambda *a: body).complete(PACKET)
    assert parsed == {"schema": "phase5-response-v1"}


def test_timeout_is_fail_closed():
    import httpx

    def boom(*a):
        raise httpx.ConnectTimeout("down")
    with pytest.raises(SynthesisError): _client(boom).complete(PACKET)


def test_http_error_is_fail_closed():
    with pytest.raises(SynthesisError): _client(lambda *a: (_ for _ in ()).throw(RuntimeError("x"))).complete(PACKET)


def test_malformed_body_is_fail_closed():
    with pytest.raises(SynthesisError): _client(lambda *a: {"no": "choices"}).complete(PACKET)
    with pytest.raises(SynthesisError): _client(lambda *a: {"choices": []}).complete(PACKET)
    with pytest.raises(SynthesisError):
        _client(lambda *a: {"choices": [{"message": {"content": "not json"}}]}).complete(PACKET)
    with pytest.raises(SynthesisError):
        _client(lambda *a: {"choices": [{"message": {"content": ""}}]}).complete(PACKET)
