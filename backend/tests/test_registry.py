import pytest

from app.parsers import (
    FluorescenceParser,
    NMRSpectrumParser,
    ParserRegistry,
    UVVisParser,
)
from app.parsers.base import BaseParser

DATAEXAMLE_DIR = __import__("pathlib").Path(__file__).parent.parent.parent / "dataexample"


class TestParserRegistry:
    def test_list_parsers(self):
        parsers = ParserRegistry.list_parsers()
        assert "nmr" in parsers
        assert "uvvis" in parsers
        assert "fluorescence" in parsers

    def test_get_parser(self):
        assert ParserRegistry.get("nmr") is NMRSpectrumParser
        assert ParserRegistry.get("uvvis") is UVVisParser
        assert ParserRegistry.get("fluorescence") is FluorescenceParser

    def test_get_unknown(self):
        with pytest.raises(KeyError):
            ParserRegistry.get("unknown")

    def test_detect_nmr(self):
        result = ParserRegistry.detect(DATAEXAMLE_DIR / "HNMRexample")
        assert result == "nmr"

    def test_detect_uvvis(self):
        result = ParserRegistry.detect(DATAEXAMLE_DIR / "紫外example" / "LA.txt")
        assert result == "uvvis"

    def test_detect_fluorescence(self):
        result = ParserRegistry.detect(DATAEXAMLE_DIR / "荧光example" / "em-lao(FDS).DX")
        assert result == "fluorescence"

    def test_detect_unknown(self):
        result = ParserRegistry.detect(DATAEXAMLE_DIR / "nonexistent.txt")
        assert result is None


class TestParserAbstract:
    def test_base_parser_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseParser()

    def test_parser_subclass(self):
        assert issubclass(NMRSpectrumParser, BaseParser)
        assert issubclass(UVVisParser, BaseParser)
        assert issubclass(FluorescenceParser, BaseParser)
