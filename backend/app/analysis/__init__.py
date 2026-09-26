from app.analysis.base import BaseAnalyzer
from app.analysis.registry import AnalyzerRegistry
from app.analysis.models import (
    AnalysisResult,
    NMRAnalysisResult,
    UVVisAnalysisResult,
    FluorescenceAnalysisResult,
)
from app.analysis.nmr_analysis import NMRAnalyzer
from app.analysis.uvvis_analysis import UVVisAnalyzer
from app.analysis.fluorescence_analysis import FluorescenceAnalyzer
from app.analysis.xrd_analysis import XRDAnalyzer
from app.analysis.hplc_analysis import HPLCAnalyzer
from app.analysis.electrochem_analysis import ElectrochemAnalyzer

AnalyzerRegistry.register("nmr", NMRAnalyzer)
AnalyzerRegistry.register("uvvis", UVVisAnalyzer)
AnalyzerRegistry.register("fluorescence", FluorescenceAnalyzer)
AnalyzerRegistry.register("xrd", XRDAnalyzer)
AnalyzerRegistry.register("hplc", HPLCAnalyzer)
AnalyzerRegistry.register("electrochem", ElectrochemAnalyzer)

__all__ = [
    "BaseAnalyzer",
    "AnalyzerRegistry",
    "AnalysisResult",
    "NMRAnalysisResult",
    "UVVisAnalysisResult",
    "FluorescenceAnalysisResult",
    "NMRAnalyzer",
    "UVVisAnalyzer",
    "FluorescenceAnalyzer",
    "XRDAnalyzer",
]
