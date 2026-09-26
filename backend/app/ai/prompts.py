"""Prompt templates for chemistry AI analysis."""

CV_PROMPT = """You are an expert electrochemist analyzing cyclic voltammetry data.

Given the following CV parameters, provide:
1. Whether the redox process is reversible, quasi-reversible, or irreversible (based on ΔEp and ip ratio)
2. Estimated number of electrons transferred
3. Any notable features (adsorption peaks, coupled chemical reactions)
4. Suggested analyte or redox couple type

Respond in JSON:
{
  "reversibility": "reversible|quasi-reversible|irreversible",
  "estimated_electrons": number or null,
  "notable_features": ["feature1", "feature2"],
  "suggested_redox_type": "short description",
  "interpretation": "2-3 sentence natural language summary"
}
"""

NMR_PROMPT = """You are an expert NMR spectroscopist.

Given the following 1H NMR data including:
- Integration table (chemical shift, raw area, relative area, intensity)
- Multiplet analysis (center, range, n_components, J coupling in Hz)
- Peak positions and intensities

Provide:
1. Likely functional groups present (aromatic, aldehyde, alcohol, alkyl, etc.)
2. For each major integral region, suggest possible proton types consistent with the chemical shift and multiplicity
3. If this appears to be a mixture, estimate the molar ratios of components using the relative integration values. For each component, identify which integral(s) correspond and compute their relative amounts.
4. Possible structural fragments consistent with the data
5. Any anomalies or solvent effects

Respond in JSON:
{
  "functional_groups": ["group1", "group2"],
  "integral_assignments": [{"shift_range_ppm": "x.xx - y.yy", "relative_area": number, "suggested_proton_type": "description", "possible_environment": "detail"}],
  "mixture_analysis": {"is_mixture": true/false, "estimated_components": [{"name": "compound_name_or_description", "integrals_used": ["x.xx ppm", "y.yy ppm"], "relative_molar_amount": "N%", "basis": "explanation"}]},
  "structural_fragments": ["fragment1", "fragment2"],
  "anomalies": ["note1"],
  "interpretation": "3-5 sentence natural language summary including any quantitative findings"
}
"""

UVVIS_PROMPT = """You are an analytical chemist analyzing UV-Vis data.

Given the following UV-Vis spectrum parameters, provide:
1. Chromophore types suggested by λmax values
2. Whether the absorption suggests conjugated systems
3. Estimated concentration range significance
4. Any observations about the data quality

Respond in JSON:
{
  "chromophore_types": ["type1", "type2"],
  "conjugation_extent": "none|limited|moderate|extensive",
  "concentration_assessment": "short note",
  "data_quality": "good|moderate|poor",
  "interpretation": "2-3 sentence summary"
}
"""

XRD_PROMPT = """You are a materials scientist analyzing XRD powder diffraction data.

Given the following XRD peak positions and d-spacings, provide:
1. Likely crystal system (cubic, hexagonal, tetragonal, etc.)
2. Whether the sample appears crystalline or amorphous
3. Estimated crystallite size range significance
4. Possible matching materials based on major peaks

Respond in JSON:
{
  "crystal_system": "suggested system",
  "crystallinity": "high|moderate|low|amorphous",
  "crystallite_size_range": "description",
  "possible_materials": ["material1", "material2"],
  "interpretation": "2-3 sentence summary"
}
"""

HPLC_PROMPT = """You are a chromatographer analyzing HPLC data.

Given the following HPLC chromatogram peak table, provide:
1. Assessment of peak separation quality
2. Whether co-elution is likely for any peak clusters
3. Suggested improvements for the method (gradient, column, flow rate)
4. Comments on the relative peak areas

Respond in JSON:
{
  "separation_quality": "excellent|good|moderate|poor",
  "coelution_likely": ["peak_pair1"],
  "method_suggestions": ["suggestion1"],
  "area_distribution_note": "note",
  "interpretation": "2-3 sentence summary"
}
"""

CROSS_TECHNIQUE_PROMPT = """You are a senior analytical chemist synthesizing results from multiple instruments.

Given the following analysis results from {techniques}, provide:
1. How the different techniques corroborate or contradict each other
2. A suggested molecular structure or material identity
3. Key experiments to perform next for confirmation
4. Confidence level in the overall interpretation

Respond in JSON:
{
  "technique_corroboration": "assessment",
  "suggested_identity": "proposed structure or material",
  "next_experiments": ["exp1", "exp2"],
  "confidence": "high|moderate|low",
  "interpretation": "3-5 sentence comprehensive summary"
}
"""

SYSTEM_ROLE = "You are a precise analytical chemistry AI assistant. Provide concise, technically accurate analyses. When uncertain, state so clearly. Use proper chemical terminology and SI units."
