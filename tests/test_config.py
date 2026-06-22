from config import Settings


def test_safe_mongo_uri_encodes_standard_scheme():
    s = Settings(mongo_uri="mongodb://user:p@ssw0rd@host:27017")
    out = s.safe_mongo_uri
    assert out.startswith("mongodb://")
    assert "p%40ssw0rd" in out  # '@' in the password is URL-encoded


def test_safe_mongo_uri_handles_srv_scheme():
    # Regression: the +srv (Atlas) scheme was previously not matched, so
    # credentials were never encoded.
    s = Settings(mongo_uri="mongodb+srv://admin:p@ss@cluster0.abc.mongodb.net")
    out = s.safe_mongo_uri
    assert out.startswith("mongodb+srv://")
    assert "p%40ss" in out


def test_safe_mongo_uri_passthrough_without_credentials():
    s = Settings(mongo_uri="mongodb://localhost:27017")
    assert s.safe_mongo_uri == "mongodb://localhost:27017"


def test_safe_mongo_uri_does_not_double_encode():
    # Regression: an already-encoded password must stay %40, not become %2540.
    s = Settings(mongo_uri="mongodb+srv://admin:p%40ss@cluster0.abc.mongodb.net")
    out = s.safe_mongo_uri
    assert "%40" in out and "%2540" not in out
