# Tests of the visitor address: the proxy chain is read from the right, because every proxy appends and only the caller writes the left side
from types import SimpleNamespace  # a request stand-in with headers and a socket address
from gpt2_demo_service.web_api.visitor_address import find_visitor_address  # code under test


def test_proxy_chain_is_read_from_the_right():  # the caller writes the left side only
    build = lambda header, host="10.1.2.3": SimpleNamespace(headers={"x-forwarded-for": header} if header is not None else {}, client=SimpleNamespace(host=host))  # fake request
    assert find_visitor_address(build("1.2.3.4, 203.0.113.9")) == "203.0.113.9"  # rightmost public entry
    assert find_visitor_address(build("1.2.3.4, 203.0.113.9, 10.0.0.7")) == "203.0.113.9"  # internal hops on the right are skipped
    assert find_visitor_address(build("1.2.3.4, 127.0.0.1, 192.168.1.5, 172.20.0.1, fd00::1")) == "1.2.3.4"  # loopback, private and unique-local hops are skipped
    assert find_visitor_address(build("2001:db8::1, 2001:db8::2")) == "2001:db8::2"  # IPv6 works the same way
    assert find_visitor_address(build("10.0.0.1, 192.168.0.9")) == "10.1.2.3" and find_visitor_address(build(None)) == "10.1.2.3"  # no public entry or no header: the socket address
    assert find_visitor_address(build("1.2.3.4, not-an-address")) == "10.1.2.3"  # a malformed last entry is never trusted past
    assert find_visitor_address(build(" 203.0.113.9 ")) == "203.0.113.9"  # spaces are ignored
