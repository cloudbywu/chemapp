from app.parsers.base import BaseParser
from app.parsers.electrochem_parser import ElectrochemParser
from app.parsers.fluorescence_parser import FluorescenceParser
from app.parsers.hplc_parser import HPLCParser
from app.parsers.jeol_jdf_parser import JEOLJDFParser
from app.parsers.nmr_jcamp_parser import NMRJCAMPParser, inspect_jcamp_numeric
from app.parsers.nmr_parser import NMRSpectrumParser
from app.parsers.registry import ParserRegistry
from app.parsers.uvvis_parser import UVVisParser
from app.parsers.xrd_parser import XRDSParser

ParserRegistry.register("nmr", NMRSpectrumParser)
ParserRegistry.register("nmr_jcamp", NMRJCAMPParser)
ParserRegistry.register("jeol_jdf", JEOLJDFParser)
ParserRegistry.register("uvvis", UVVisParser)
ParserRegistry.register("fluorescence", FluorescenceParser)
ParserRegistry.register("xrd", XRDSParser)
ParserRegistry.register("hplc", HPLCParser)
ParserRegistry.register("electrochem", ElectrochemParser)

__all__ = [
    "BaseParser",
    "FluorescenceParser",
    "JEOLJDFParser",
    "NMRJCAMPParser",
    "NMRSpectrumParser",
    "ParserRegistry",
    "UVVisParser",
    "XRDSParser",
    "inspect_jcamp_numeric",
]
