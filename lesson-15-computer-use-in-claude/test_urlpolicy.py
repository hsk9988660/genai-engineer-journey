import pytest

from urlpolicy import BlockedURL, DomainPolicy, resolve, safe_display

P = DomainPolicy(["stays.example"])


def test_the_allowed_host_and_its_subdomains_pass_and_the_url_is_returned_unchanged():
    for url in ("https://stays.example/", "https://www.stays.example/rooms/1?x=1", "https://STAYS.EXAMPLE/a", "https://stays.example./a", "https://stays.example:443/a"):
        assert P.check(url) == url


@pytest.mark.parametrize("url,why", [
    ("https://stays.example.evil.example/", "allow-list"),  # the allowed name used as a PREFIX of the real host
    ("https://evilstays.example/", "allow-list"),  # the allowed name used as a suffix without a dot
    ("https://stays.example@evil.example/", "credentials"),  # userinfo trick
    ("https://evil.example\\@stays.example/", "backslash"),  # a backslash browsers read as a slash
    ("https://stays.example\\.evil.example/", "backslash"),
    ("http://stays.example/", "scheme"),
    ("javascript:alert(1)", "scheme"),
    ("data:text/html,<script>1</script>", "scheme"),
    ("file:///etc/passwd", "scheme"),
    ("ftp://stays.example/", "scheme"),
    ("https://127.0.0.1/", "IP"),
    ("https://10.0.0.5/", "IP"),
    ("https://[::1]/", "IP"),
    ("https://xn--stays-9ya.example/", "punycode"),
    ("https://stäys.example/", "non-ASCII"),
    ("https://stays.example:8443/", "port"),
    ("https://stays.example/ a", "whitespace"),
    ("https://stays.example/\x00", "control"),
    ("https:///nohost", "no host"),
    ("", "empty"),
    ("https://" + "a" * 2100 + ".example/", "oversized"),
])
def test_known_url_tricks_are_blocked(url, why):
    with pytest.raises(BlockedURL):
        P.check(url)
    assert P.allowed_url(url) is False


def test_a_non_string_is_blocked():
    for bad in (None, 5, b"https://stays.example/"):
        with pytest.raises(BlockedURL):
            P.check(bad)


def test_http_and_port_80_are_allowed_only_if_asked_for():
    lax = DomainPolicy(["stays.example"], allow_http=True)
    assert lax.allowed_url("http://stays.example/") and lax.allowed_url("http://stays.example:80/")
    assert not lax.allowed_url("http://stays.example:8080/") and not P.allowed_url("http://stays.example:80/")


def test_at_least_one_host_is_required_and_several_may_be_given():
    with pytest.raises(ValueError):
        DomainPolicy([])
    two = DomainPolicy(["a.example", "b.example"])
    assert two.allowed_url("https://x.a.example/") and two.allowed_url("https://b.example/") and not two.allowed_url("https://c.example/")


def test_a_relative_link_is_resolved_before_it_is_checked_so_a_protocol_relative_one_cannot_smuggle_the_agent_off_site():
    base = "https://stays.example/s/stockholm"
    assert resolve(base, "/rooms/1") == "https://stays.example/rooms/1"
    sneaky = resolve(base, "//evil.example/x")
    assert sneaky == "https://evil.example/x" and not P.allowed_url(sneaky)


def test_displayed_urls_show_host_and_path_but_hide_the_query_and_fragment():
    assert safe_display("https://stays.example/rooms/1?token=SECRET#frag") == "https://stays.example/rooms/1?…"
    assert "SECRET" not in safe_display("https://evil.example/c?d=SECRET")
    assert safe_display("https://stays.example/a") == "https://stays.example/a"


def test_a_credentials_prefix_is_refused_even_when_the_real_host_is_the_allowed_one():
    with pytest.raises(BlockedURL, match="credentials"):
        P.check("https://evil.example@stays.example/")  # the browser would go to stays.example, but a user could be misled about it


def test_ip_addresses_and_look_alike_hosts_are_blocked_even_if_someone_puts_them_on_the_allow_list():
    for host, url in (("127.0.0.1", "https://127.0.0.1/"), ("xn--stays-9ya.example", "https://xn--stays-9ya.example/"), ("stäys.example", "https://stäys.example/")):
        with pytest.raises(BlockedURL):
            DomainPolicy([host]).check(url)


def test_an_oversized_url_on_an_allowed_host_is_blocked():
    with pytest.raises(BlockedURL, match="oversized"):
        P.check("https://stays.example/" + "a" * 2100)
