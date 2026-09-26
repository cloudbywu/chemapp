import numpy as np

from app.core.models import NMRNucleus, Peak, SampleInfo, Spectrum, Technique


class TestPeak:
    def test_peak_creation(self):
        p = Peak(position=7.26, intensity=1000.0, area=500.0, assignment="CHCl3")
        assert p.position == 7.26
        assert p.intensity == 1000.0
        assert p.area == 500.0
        assert p.assignment == "CHCl3"
        assert p.width is None
        assert p.multiplicity == ""

    def test_peak_defaults(self):
        p = Peak(position=3.5, intensity=0.5)
        assert p.area is None
        assert p.coupling_constant is None


class TestSampleInfo:
    def test_defaults(self):
        s = SampleInfo()
        assert s.name == ""
        assert s.solvent == ""
        assert s.concentration is None

    def test_custom(self):
        s = SampleInfo(name="test", solvent="CDCl3", concentration=10.0, concentration_unit="mg/mL")
        assert s.name == "test"
        assert s.solvent == "CDCl3"
        assert s.concentration == 10.0


class TestSpectrum:
    def test_creation(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        spec = Spectrum(
            technique=Technique.UVVIS,
            x_data=x,
            y_data=y,
            x_label="Wavelength",
            y_label="Absorbance",
            x_unit="nm",
            y_unit="abs",
        )
        assert spec.technique == Technique.UVVIS
        assert spec.num_points == 100
        assert spec.x_range == (0.0, 10.0)
        assert len(spec.peaks) == 0

    def test_to_dict_from_dict(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.1, 0.5, 0.2])
        spec = Spectrum(
            technique=Technique.NMR,
            x_data=x,
            y_data=y,
            x_label="ppm",
            y_label="intensity",
            x_unit="ppm",
            y_unit="arb.",
            parameters={"nucleus": "1H"},
            metadata=SampleInfo(name="sample1", solvent="CDCl3"),
            peaks=[Peak(position=2.0, intensity=0.5, area=0.3)],
            source_file="test.fid",
        )
        d = spec.to_dict()
        restored = Spectrum.from_dict(d)
        assert restored.technique == spec.technique
        assert np.allclose(restored.x_data, spec.x_data)
        assert np.allclose(restored.y_data, spec.y_data)
        assert restored.metadata.name == "sample1"
        assert restored.metadata.solvent == "CDCl3"
        assert len(restored.peaks) == 1
        assert restored.peaks[0].position == 2.0

    def test_repr(self):
        spec = Spectrum(
            technique=Technique.FLUORESCENCE,
            x_data=np.linspace(200, 800, 3001),
            y_data=np.zeros(3001),
        )
        r = repr(spec)
        assert "Fluorescence" in r
        assert "3001" in r


class TestTechnique:
    def test_technique_values(self):
        assert Technique.NMR.value == "NMR"
        assert Technique.UVVIS.value == "UV-Vis"
        assert Technique.FLUORESCENCE.value == "Fluorescence"


class TestNMRNucleus:
    def test_nucleus_values(self):
        assert NMRNucleus.H1.value == "1H"
        assert NMRNucleus.C13.value == "13C"
