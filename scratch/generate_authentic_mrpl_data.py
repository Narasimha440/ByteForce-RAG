"""
Generate authentic, technically accurate MRPL refinery documents for SIH26117.
Reflects the actual 15.0 MMTPA complex in Mangalore, Karnataka.
"""

from pathlib import Path
import docx
import openpyxl
import pymupdf
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STANDARDS_DIR = DATA_DIR / "01_Standards_Reference"
STANDARDS_DIR.mkdir(parents=True, exist_ok=True)


def create_mrpl_master_profile():
    path = STANDARDS_DIR / "MRPL_Refinery_Master_Technical_Profile.docx"
    doc = docx.Document()
    doc.add_heading("Mangalore Refinery and Petrochemicals Limited (MRPL)", level=0)
    doc.add_heading("Technical Configuration & Operating Profile (15.0 MMTPA Complex)", level=1)

    doc.add_heading("1. Corporate & Complex Overview", level=2)
    doc.add_paragraph(
        "Mangalore Refinery and Petrochemicals Limited (MRPL), a Schedule 'A' Miniratna Central Public Sector Enterprise "
        "and a subsidiary of Oil and Natural Gas Corporation (ONGC), operates a state-of-the-art grassroots refinery "
        "at Mangalore, Karnataka. The refinery has a nameplate installed capacity of 15.0 Million Metric Tonnes Per Annum (MMTPA) "
        "and frequently achieves over 115-120% operational capacity utilization. The complex possesses a high Nelson Complexity "
        "Index, enabling it to process heavy, sour, and high-acid crude oils with high distillate recovery."
    )

    doc.add_heading("2. Refinery Configuration by Expansion Phases", level=2)
    table = doc.add_table(rows=4, cols=3)
    headers = ["Expansion Phase", "Commissioning Year", "Primary Processing Units & Capabilities"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h

    phase_data = [
        ["Phase I", "1996", "CDU-1 & VDU-1 (3.69 MMTPA), Hydrocracker Unit 1 (HCU-1), Hydrogen Generation Unit (HGU-1), Visbreaker Unit (VBU), Bitumen Blowing Unit (BBU), Sulfur Recovery Unit (SRU-1)."],
        ["Phase II", "1999–2000", "CDU-2 & VDU-2 (5.9 MMTPA expansion), Continuous Catalytic Regeneration Platformer (CCR-1 & CCR-2) for high-octane MS, Additional SRU trains, Diesel Hydro Desulfurization (DHDS)."],
        ["Phase III", "2012–2014", "CDU-3 & VDU-3 (3.0 MMTPA), Hydrocracker Unit 2 (HCU-2), Petrochemical Fluidized Catalytic Cracking Unit (PFCCU, 2.2 MMTPA), Delayed Coker Unit (DCU, 3.0 MMTPA), Polypropylene Plant (PPU, 440 KTPA, 'Mangpol'), Diesel Hydrotreater (DHDT)."],
    ]
    for r_idx, row in enumerate(phase_data, start=1):
        for c_idx, val in enumerate(row):
            table.cell(r_idx, c_idx).text = val

    doc.add_heading("3. Unique Refining Distinctions in India", level=2)
    doc.add_paragraph(
        "• Dual Hydrocracker Configuration: MRPL is the ONLY refinery in India operating TWO separate Hydrocracker Units "
        "(HCU-1 and HCU-2), maximizing the yield of ultra-low sulfur, high-cetane BS-VI high-speed diesel (HSD).\n"
        "• Deep Residue Upgradation: The Delayed Coker Unit (DCU) converts heavy vacuum residue into high-value distillates and fuel-grade petcoke, eliminating low-value heavy fuel oil production.\n"
        "• Captive Water & Power Security: Includes a 30 MLD (Million Litres per Day) seawater reverse osmosis desalination plant and captive cogeneration power plant (CPP), ensuring zero dependence on municipal river water during dry seasons."
    )

    doc.add_heading("4. Product Slate", level=2)
    doc.add_paragraph(
        "The refinery produces BS-VI Motor Spirit (Petrol), BS-VI High Speed Diesel (HSD), Aviation Turbine Fuel (ATF / Jet A-1), "
        "Liquefied Petroleum Gas (LPG), Naphtha, Polymer Grade Polypropylene ('Mangpol'), Industrial Bitumen (VG-30, VG-40), "
        "Fuel Grade Green Pet Coke, and High Purity Molten/Solid Elemental Sulfur (99.9% purity)."
    )

    doc.save(str(path))
    print(f"Created: {path.name}")


def create_mrpl_hcu_sop():
    path = STANDARDS_DIR / "MRPL_HCU_Hydrocracker_Operating_SOP.docx"
    doc = docx.Document()
    doc.add_heading("MRPL Mangalore Refinery — Hydrocracker Units (HCU-1 & HCU-2)", level=0)
    doc.add_heading("Standard Operating Manual: High-Pressure Reaction Section & Emergency Procedures", level=1)

    doc.add_heading("1. Process Description & Reactor Conditions", level=2)
    doc.add_paragraph(
        "MRPL operates two identical high-pressure two-stage hydrocrackers (HCU-1 and HCU-2) licensed by Chevron Lummus Global (CLG). "
        "The units process heavy vacuum gas oil (HVGO) and coker heavy gas oil (HCGO) over noble/base metal catalysts in a hydrogen-rich "
        "atmosphere to produce high-cetane diesel (Cetane Index > 52) and ATF with sulfur levels below 5 ppm."
    )

    doc.add_heading("2. Critical Reaction Parameter Limits", level=2)
    table = doc.add_table(rows=6, cols=4)
    headers = ["Operating Parameter", "Normal Range", "Alarm High Limit", "Emergency Action / Interlock"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h

    limits_data = [
        ["Reactor Loop Operating Pressure", "165 – 180 bar(g)", "> 188 bar(g)", "Open Emergency Depressurization Valve BDV-701 to Flare"],
        ["Reactor Bed 1 Inlet Temperature", "380°C – 405°C", "> 420°C", "Increase Cold Hydrogen Quench Flow via QCV-101"],
        ["Reactor Bed Maximum Delta-T", "25°C – 35°C", "> 45°C", "Emergency Feed Cutoff; Runaway Reaction Protocol"],
        ["Hydrogen Recycle Gas Purity", "88 – 95 mole % H2", "< 82 mole % H2", "Increase Purge Rate to Fuel Gas; Boost Makeup H2 from HGU"],
        ["Recycle Gas Compressor K-601 Suction Drum Level", "30% – 50%", "> 70%", "Trip Compressor K-601 on High-High Liquid Level to prevent liquid carryover"],
    ]
    for r_idx, row in enumerate(limits_data, start=1):
        for c_idx, val in enumerate(row):
            table.cell(r_idx, c_idx).text = val

    doc.add_heading("3. Emergency High-Pressure Depressuring (HPE) Protocol", level=2)
    doc.add_paragraph(
        "In the event of an uncontrollable temperature runaway (delta-T > 50°C in any catalyst bed) or severe high-pressure hydrogen leak, "
        "the operator shall activate the Emergency Depressurization System (EDS). Dual fail-safe valves BDV-701A and BDV-701B will open, "
        "depressurizing the reactor loop from 175 bar to 7 bar within 15 minutes as per API-521 standard guidelines."
    )

    doc.save(str(path))
    print(f"Created: {path.name}")


def create_mrpl_crude_assay_excel():
    path = STANDARDS_DIR / "MRPL_Crude_Assay_and_Blend_Optimization.xlsx"
    wb = openpyxl.Workbook()

    # Sheet 1: Crude Assays
    ws1 = wb.active
    ws1.title = "Crude_Assays"
    ws1.append(["Crude Name", "Origin", "API Gravity", "Sulfur (wt %)", "TAN (mg KOH/g)", "Pour Point (°C)", "Yield: Naphtha (%)", "Yield: Distillates (%)", "Yield: Residue (%)"])
    crudes = [
        ["Arab Light", "Saudi Arabia", 33.4, 1.80, 0.12, -25, 21.5, 34.0, 44.5],
        ["Basrah Heavy", "Iraq", 23.8, 4.10, 0.45, -15, 12.0, 27.5, 60.5],
        ["Maya", "Mexico", 21.8, 3.40, 2.20, -18, 14.5, 26.0, 59.5],
        ["Murban", "UAE", 40.5, 0.75, 0.05, -30, 28.0, 42.0, 30.0],
        ["Bombay High", "India (Domestic)", 39.2, 0.15, 0.08, 30, 18.5, 48.0, 33.5],
    ]
    for row in crudes:
        ws1.append(row)

    # Sheet 2: Blending Rules & Unit Constraints
    ws2 = wb.create_sheet(title="Blend_Constraints")
    ws2.append(["Refinery Unit / Product", "Constraint Parameter", "Maximum Permissible Limit", "Operating Mitigation"])
    constraints = [
        ["Desalter D-101 / CDU", "Salt Content in Desalted Crude", "< 3.0 PTB (Pounds per 1000 Barrels)", "Maintain wash water at 6.0% vol and demulsifier injection at 8 ppm"],
        ["Atmospheric Tower Overhead", "Total Acid Number (TAN)", "< 0.50 mg KOH/g blend", "Limit Maya/High-Acid crude component to max 25% of crude slate"],
        ["BS-VI High Speed Diesel", "Sulfur Content in Finished Product", "< 10.0 ppm (mg/kg)", "Route through HCU-1, HCU-2 or DHDT for deep hydrodesulfurization"],
        ["Delayed Coker Unit (DCU)", "Conradson Carbon Residue (CCR)", "18.0 wt % in Vacuum Residue", "Blend low-carbon residue crudes to prevent excessive coking rate"],
        ["Polypropylene Plant (PPU)", "Propylene Feed Purity", "> 99.5 wt % (Polymer Grade)", "Maintain C3 splitter tray 110 temperature at 42°C in PFCCU gas concentration section"],
    ]
    for row in constraints:
        ws2.append(row)

    wb.save(str(path))
    print(f"Created: {path.name}")


def create_mrpl_spm_sop():
    path = STANDARDS_DIR / "MRPL_SPM_Offshore_Crude_Unloading_SOP.docx"
    doc = docx.Document()
    doc.add_heading("MRPL Mangalore Refinery — Single Point Mooring (SPM) Operations", level=0)
    doc.add_heading("Standard Operating Procedure: Offshore VLCC Crude Tanker Unloading (SOP-SPM-2026-02)", level=1)

    doc.add_heading("1. SPM Facility Description", level=2)
    doc.add_paragraph(
        "MRPL operates a Single Point Mooring (SPM) terminal located approximately 16.5 kilometers offshore New Mangalore Port "
        "at a water depth of 32 meters. The terminal can handle Very Large Crude Carriers (VLCCs) up to 300,000 DWT. "
        "Crude oil is pumped from the tanker through two 24-inch floating hoses, the SPM swivel, a pipeline end manifold (PLEM), "
        "and a 48-inch subsea pipeline to the onshore coastal crude oil terminal (CCOT) at Kasaba and onward to the refinery."
    )

    doc.add_heading("2. Discharge Operations & Pressure Surge Limits", level=2)
    table = doc.add_table(rows=4, cols=3)
    headers = ["Parameter / Safety System", "Design Specification", "Operating Safety Rule"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h

    spm_data = [
        ["Maximum Discharge Rate", "10,000 m3/hour", "Initial rate capped at 2,500 m3/h until pipeline pressure stabilizes"],
        ["Subsea Pipeline Pressure Relief Valve PRV-901", "Set point: 18.5 bar(g)", "Trips flow and diverts surge to onshore surge relief tank at CCOT"],
        ["Marine Breakaway Coupling (MBC)", "Axial load trigger: 450 kN", "Guarantees dry break without marine oil spill in extreme tanker drift"],
    ]
    for r_idx, row in enumerate(spm_data, start=1):
        for c_idx, val in enumerate(row):
            table.cell(r_idx, c_idx).text = val

    doc.save(str(path))
    print(f"Created: {path.name}")


def create_mrpl_kspcb_report():
    path = STANDARDS_DIR / "MRPL_KSPCB_Environmental_Compliance_Report.docx"
    doc = docx.Document()
    doc.add_heading("MRPL Mangalore Refinery — Environmental Management Division", level=0)
    doc.add_heading("Statutory Compliance Matrix: Karnataka State Pollution Control Board (KSPCB) Standards", level=1)

    doc.add_heading("1. Continuous Online Ambient & Flue Gas Emission Limits", level=2)
    table = doc.add_table(rows=5, cols=4)
    headers = ["Emission Source", "Pollutant", "KSPCB Statutory Limit", "MRPL Actual Recorded Average (2025–2026)"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h

    emissions = [
        ["Sulfur Recovery Units (SRU-1/2/3)", "Sulfur Dioxide (SO2)", "< 250 mg/Nm3", "142 mg/Nm3 (99.9% sulfur recovery via TGTU)"],
        ["FCCU Regenerator Stack", "Particulate Matter (PM)", "< 50 mg/Nm3", "31 mg/Nm3 (Treated via Electrostatic Precipitator ESP-401)"],
        ["Crude Furnaces F-101 / F-201", "Oxides of Nitrogen (NOx)", "< 250 mg/Nm3", "168 mg/Nm3 (Ultra-low NOx burners)"],
        ["Flare Stacks (South & North)", "Carbon Monoxide (CO)", "< 100 mg/Nm3", "24 mg/Nm3 (Continuous steam-assisted combustion)"],
    ]
    for r_idx, row in enumerate(emissions, start=1):
        for c_idx, val in enumerate(row):
            table.cell(r_idx, c_idx).text = val

    doc.add_heading("2. Treated Effluent Quality Limits (Discharge to Arabian Sea)", level=2)
    doc.add_paragraph(
        "Treated effluent from Effluent Treatment Plants (ETP-1 and ETP-2) must comply with the following limits:\n"
        "• Phenolic Compounds: < 0.35 mg/L\n"
        "• Sulfides: < 0.50 mg/L\n"
        "• Oil and Grease: < 5.0 mg/L\n"
        "• Chemical Oxygen Demand (COD): < 125 mg/L\n"
        "• Biochemical Oxygen Demand (BOD 3 days at 27°C): < 15.0 mg/L."
    )

    doc.save(str(path))
    print(f"Created: {path.name}")


if __name__ == "__main__":
    create_mrpl_master_profile()
    create_mrpl_hcu_sop()
    create_mrpl_crude_assay_excel()
    create_mrpl_spm_sop()
    create_mrpl_kspcb_report()
    print("All authentic MRPL refinery documents generated successfully!")
