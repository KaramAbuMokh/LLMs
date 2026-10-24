# Finds the caller's address behind the Hugging Face proxy, for the per-visitor rate limit
import ipaddress  # parses and classifies addresses
from fastapi import Request  # incoming request type

INTERNAL_NETWORKS = tuple(ipaddress.ip_network(text) for text in (  # hops inside the proxy chain, never a visitor (documentation ranges stay public on purpose, unlike ipaddress.is_private)
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",  # IPv4 private, shared, loopback and link-local
    "fc00::/7", "::1/128", "fe80::/10",  # IPv6 unique-local, loopback and link-local
))


def is_internal_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:  # True for a private or loopback proxy hop
    return any(address.version == network.version and address in network for network in INTERNAL_NETWORKS)  # inside any internal range


def find_visitor_address(http_request: Request) -> str:  # the caller's address: the rightmost public entry of the proxy header, else the socket address
    socket_address = http_request.client.host if http_request.client else "unknown"  # what the server itself sees
    forwarded_entries = http_request.headers.get("x-forwarded-for", "").split(",")  # every proxy appends the address it received from, so the caller can only write the LEFT side
    for entry in reversed(forwarded_entries):  # walk from the proxy-written end
        try:  # an entry that is not an address stops the walk
            address = ipaddress.ip_address(entry.strip())  # parsed address
        except ValueError:  # empty or malformed: nothing after it can be trusted
            return socket_address  # fall back to the socket
        if not is_internal_address(address):  # first public hop from the right is the real caller
            return str(address)  # normalised text, so one address has one spelling
    return socket_address  # no header, or only internal hops
