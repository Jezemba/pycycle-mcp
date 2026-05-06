"""OVS — Output Verification System checks for pyCycle CPACS output.

Validates that the pyCycle adapter writes expected XPaths with plausible values.
Self-contained: no cross-repo dependencies.
"""

from xml.etree import ElementTree as ET

SAMPLE_PYCYCLE_OUTPUT = """\
<?xml version="1.0"?>
<cpacs>
  <vehicles>
    <aircraft>
      <model uID="test">
        <name>OVS Test Aircraft</name>
      </model>
    </aircraft>
    <engines>
      <engine>
        <analysis>
          <mcpResults>
            <solver>pycycle_openmdao</solver>
            <TSFC_lb_lbf_hr>0.885</TSFC_lb_lbf_hr>
            <Fn_N>26528.4</Fn_N>
            <OPR>30.55</OPR>
            <BPR>1.5</BPR>
            <fuelFlow_kg_s>0.665</fuelFlow_kg_s>
          </mcpResults>
        </analysis>
      </engine>
    </engines>
  </vehicles>
</cpacs>
"""


def test_pycycle_output_structure():
    root = ET.fromstring(SAMPLE_PYCYCLE_OUTPUT)
    assert root.tag == "cpacs"
    assert root.find(".//vehicles/aircraft") is not None


def test_pycycle_results_present():
    root = ET.fromstring(SAMPLE_PYCYCLE_OUTPUT)
    mcp_res = root.find(".//vehicles/engines/engine/analysis/mcpResults")
    assert mcp_res is not None


def test_pycycle_solver_tag():
    root = ET.fromstring(SAMPLE_PYCYCLE_OUTPUT)
    solver = root.find(".//mcpResults/solver")
    assert solver is not None and solver.text == "pycycle_openmdao"


def test_pycycle_tsfc_range():
    root = ET.fromstring(SAMPLE_PYCYCLE_OUTPUT)
    el = root.find(".//mcpResults/TSFC_lb_lbf_hr")
    assert el is not None and el.text is not None
    assert 0.0 <= float(el.text) <= 5.0


def test_pycycle_thrust_range():
    root = ET.fromstring(SAMPLE_PYCYCLE_OUTPUT)
    el = root.find(".//mcpResults/Fn_N")
    assert el is not None and el.text is not None
    assert 0.0 <= float(el.text) <= 1e7


def test_pycycle_opr_range():
    root = ET.fromstring(SAMPLE_PYCYCLE_OUTPUT)
    el = root.find(".//mcpResults/OPR")
    assert el is not None and el.text is not None
    assert 1.0 <= float(el.text) <= 100.0


def test_pycycle_bpr_range():
    root = ET.fromstring(SAMPLE_PYCYCLE_OUTPUT)
    el = root.find(".//mcpResults/BPR")
    assert el is not None and el.text is not None
    assert 0.0 <= float(el.text) <= 50.0


def test_pycycle_fuel_flow_range():
    root = ET.fromstring(SAMPLE_PYCYCLE_OUTPUT)
    el = root.find(".//mcpResults/fuelFlow_kg_s")
    assert el is not None and el.text is not None
    assert 0.0 <= float(el.text) <= 100.0
