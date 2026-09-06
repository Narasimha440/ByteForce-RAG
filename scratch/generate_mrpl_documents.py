"""
Script to generate realistic MRPL refinery industrial reference documents
for SIH Problem Statement SIH26117.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import docx
import openpyxl
import pymupdf

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def create_fccu_manual():
    path = DATA_DIR / "01_Standards_Reference" / "MRPL_FCCU_Operating_Manual.docx"
    doc = docx.Document()
    doc.add_heading("MRPL Mangalore Refinery — Fluid Catalytic Cracking Unit (FCCU)", level=0)
    doc.add_heading("Standard Operating Manual & Technical Specifications (Document ID: MRPL-SOP-FCCU-2026)", level=1)

    doc.add_heading("1. Unit Overview & Process Chemistry", level=2)
    doc.add_paragraph(
        "The Fluid Catalytic Cracking Unit (FCCU) at Mangalore Refinery processes heavy vacuum gas oil (HVGO) "
        "and atmospheric residue blends into high-octane gasoline, light cycle oil (LCO), and LPG. "
        "The reaction takes place in a vertical riser reactor with zeolite catalyst particles fluidised by hydrocarbon vapors."
    )

    doc.add_heading("2. Normal Operating Limits & Critical Parameter Matrix", level=2)
    table = doc.add_table(rows=6, cols=4)
    headers = ["Parameter", "Instrument Tag", "Normal Range", "Trip Limit / Interlock"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h

    data = [
        ["Riser Reactor Outlet Temperature", "TI-201A/B", "520°C – 540°C", "> 555°C (Auto feed cut ESD-501)"],
        ["Regenerator Dense Bed Temperature", "TI-204A-D", "680°C – 715°C", "> 740°C (Cyclone metallurgy limit)"],
        ["Regenerator Cyclone Differential Pressure", "DP-401", "12 – 25 kPa", "> 35 kPa (Catalyst loss hazard)"],
        ["Spent Catalyst Slide Valve Position", "XV-301", "35% – 65% Open", "Fail Close (FC) on low air flow"],
        ["Combustion Air Blower Discharge Pressure", "PT-101", "2.8 – 3.4 bar(g)", "< 2.2 bar(g) (Trip catalyst circulation)"],
    ]
    for row_idx, row in enumerate(data, start=1):
        for col_idx, val in enumerate(row):
            table.cell(row_idx, col_idx).text = val

    doc.add_heading("3. Emergency Shutdown & Safety Instrumented System (SIS)", level=2)
    doc.add_paragraph(
        "In the event of total electrical failure or loss of combustion air, Emergency Shutdown Valve XV-301 "
        "must close within 3.0 seconds to prevent reverse flow of hydrocarbon vapors into the catalyst regenerator. "
        "All solenoid valves are powered by 24 VDC redundant uninterruptible power supply (UPS) circuits."
    )
    doc.save(str(path))
    print(f"Created: {path.name}")


def create_cdu_sop():
    path = DATA_DIR / "01_Standards_Reference" / "MRPL_CDU_VDU_Crude_Distillation_SOP.docx"
    doc = docx.Document()
    doc.add_heading("MRPL Mangalore Refinery — Crude & Vacuum Distillation Unit (CDU/VDU)", level=0)
    doc.add_heading("Standard Operating Procedure: Crude Blend Optimization & Tower Operations", level=1)

    doc.add_heading("1. Feed Optimization & Desalting", level=2)
    doc.add_paragraph(
        "The Crude Distillation Unit processes crude blends ranging from sweet light crudes (Arab Light, Bonny Light) "
        "to heavy sour crudes (Maya, Basrah Heavy). Electrostatic Desalter D-101 maintains salt content below 3.0 PTB "
        "and water cut below 0.1% vol to prevent overhead condenser fouling and naphthenic acid corrosion."
    )

    doc.add_heading("2. Column Operating Conditions & Draw Temperatures", level=2)
    table = doc.add_table(rows=5, cols=4)
    headers = ["Distillation Cut", "Column Tray / Section", "Draw Temperature", "Quality Specification"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h

    data = [
        ["Heavy Naphtha", "Tray 38-42", "140°C – 165°C", "RVP < 0.85 bar, N+2A > 55%"],
        ["Aviation Turbine Fuel (ATF / Kerosene)", "Tray 28-32", "195°C – 230°C", "Smoke Point > 21 mm, Flash > 38°C"],
        ["High Speed Diesel (HSD / Gas Oil)", "Tray 14-18", "290°C – 345°C", "Cetane Index > 48, Sulfur < 10 ppm (BS-VI)"],
        ["Atmospheric Residue (AR)", "Column Sump", "345°C – 360°C", "Feed to Vacuum Distillation Unit (VDU)"],
    ]
    for row_idx, row in enumerate(data, start=1):
        for col_idx, val in enumerate(row):
            table.cell(row_idx, col_idx).text = val

    doc.add_heading("3. Furnace F-101 Coil Protection", level=2)
    doc.add_paragraph(
        "Crude charge furnace F-101 skin temperatures must not exceed 480°C. Flow meters FT-201 through FT-204 "
        "monitor pass balance across the four radiant coils to prevent localized coke deposition."
    )
    doc.save(str(path))
    print(f"Created: {path.name}")


def create_oisd_standard():
    path = DATA_DIR / "01_Standards_Reference" / "OISD_156_Fire_Protection_Refineries.docx"
    doc = docx.Document()
    doc.add_heading("Oil Industry Safety Directorate — OISD Standard 156", level=0)
    doc.add_heading("Fire Protection & Toxic Gas Detection Facilities for Petroleum Refineries", level=1)

    doc.add_heading("1. Scope and Mandatory Provisions", level=2)
    doc.add_paragraph(
        "This standard lays down minimum requirements for fixed fire protection, hydrocarbon gas detection, "
        "and toxic gas monitoring (H2S and SO2) systems in hydrocarbon processing plants, tank farms, and LPG storage spheres."
    )

    doc.add_heading("2. Gas Detector Allocation & Alarm Thresholds", level=2)
    table = doc.add_table(rows=4, cols=4)
    headers = ["Gas Hazard", "Detector Type", "Alarm 1 (Low Alert)", "Alarm 2 (High / Auto Deluge)"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h

    data = [
        ["Flammable Hydrocarbon Vapor", "Catalytic / IR Open Path", "20% LEL (Audible beacon)", "60% LEL (Trip fuel gas & start water spray)"],
        ["Hydrogen Sulfide (H2S)", "Electrochemical Sensor", "10 ppm (Mandatory breathing gear)", "20 ppm (Evacuate unit & ESD isolation)"],
        ["Sulfur Dioxide (SO2)", "Optical / Electrochemical", "2 ppm (Warning alert)", "5 ppm (SRU unit shutdown)"],
    ]
    for row_idx, row in enumerate(data, start=1):
        for col_idx, val in enumerate(row):
            table.cell(row_idx, col_idx).text = val

    doc.save(str(path))
    print(f"Created: {path.name}")


def create_turnaround_sop():
    path = DATA_DIR / "02_Templates_Checklists_Forms" / "MRPL_Refinery_Turnaround_Shutdown_SOP.docx"
    doc = docx.Document()
    doc.add_heading("MRPL Refinery Turnaround & Major Overhaul Management Plan", level=0)
    doc.add_heading("Standard Operating Procedure: Hydrocarbon Freeing, Blinding, and Vessel Entry (SOP-TA-2026-01)", level=1)

    doc.add_heading("1. Pre-Turnaround Unit Depressurization & Blinding", level=2)
    doc.add_paragraph(
        "Prior to breaking flanges or unheading columns, all battery limit connections must be isolated with "
        "spectacle blinds or spade plates as per the Master Blind List. Blind installation must be verified by two independent engineers."
    )

    doc.add_heading("2. Confined Space Vessel Entry Criteria", level=2)
    table = doc.add_table(rows=5, cols=3)
    headers = ["Atmospheric Parameter", "Permissible Safe Entry Limit", "Testing Frequency"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h

    data = [
        ["Oxygen (O2) Concentration", "19.5% to 23.5% vol", "Continuous during entry"],
        ["Combustible Hydrocarbon Gas", "< 1.0% LEL (strictly 0.0% preferred)", "Every 2 hours or shift handover"],
        ["Hydrogen Sulfide (H2S)", "< 5.0 ppm", "Continuous portable gas monitor"],
        ["Carbon Monoxide (CO)", "< 25.0 ppm", "Before every permit re-validation"],
    ]
    for row_idx, row in enumerate(data, start=1):
        for col_idx, val in enumerate(row):
            table.cell(row_idx, col_idx).text = val

    doc.save(str(path))
    print(f"Created: {path.name}")


def create_instrumentation_excel():
    path = DATA_DIR / "04_Instrumentation_Drawings" / "MRPL_Instrumentation_Master_Index.xlsx"
    wb = openpyxl.Workbook()

    # Sheet 1: Transmitters
    ws1 = wb.active
    ws1.title = "Transmitters"
    ws1.append(["Tag No.", "Service Description", "Process Unit", "Calibrated Range", "Output Signal", "Power Supply", "P&ID Reference"])
    transmitters = [
        ["PT-101", "FCCU Air Blower Discharge Pressure", "FCCU", "0 to 5.0 bar(g)", "4-20 mA HART", "24 VDC", "MRPL-PID-FCCU-01"],
        ["PT-102", "CDU Atmospheric Tower Overhead Pressure", "CDU", "0 to 2.5 bar(g)", "4-20 mA Foundation Fieldbus", "24 VDC", "MRPL-PID-CDU-04"],
        ["FT-201", "Furnace F-101 Pass 1 Charge Flow", "CDU", "0 to 120 m3/h", "4-20 mA HART", "24 VDC", "MRPL-PID-CDU-02"],
        ["FT-204", "Furnace F-101 Pass 4 Charge Flow", "CDU", "0 to 120 m3/h", "4-20 mA HART", "24 VDC", "MRPL-PID-CDU-02"],
        ["LT-301", "Desalter D-101 Water Interface Level", "CDU", "0 to 1800 mm", "4-20 mA Guided Wave Radar", "24 VDC", "MRPL-PID-CDU-01"],
        ["TI-201", "FCCU Riser Outlet Temperature", "FCCU", "0 to 800 °C", "Duplex Type K Thermocouple", "External Transmitter", "MRPL-PID-FCCU-03"],
        ["TI-204", "Regenerator Dense Bed Temperature", "FCCU", "0 to 1000 °C", "Triple Redundant Thermocouple", "External Transmitter", "MRPL-PID-FCCU-05"],
    ]
    for row in transmitters:
        ws1.append(row)

    # Sheet 2: Valves
    ws2 = wb.create_sheet(title="Control_Valves")
    ws2.append(["Valve Tag", "Service Description", "Size & Rating", "Fail Action", "Stroke Time", "SIL Level", "Actuator Type"])
    valves = [
        ["XV-301", "FCCU Spent Catalyst Slide Valve", "36 inch Class 300", "Fail Close (FC)", "< 3.0 sec", "SIL-3", "Hydraulic / 24 VDC Solenoid"],
        ["ESD-501", "CDU Crude Charge Emergency Isolation", "16 inch Class 600", "Fail Close (FC)", "< 5.0 sec", "SIL-3", "Pneumatic Spring Return"],
        ["FCV-102", "Desalter Wash Water Flow Control", "4 inch Class 150", "Fail Open (FO)", "12 sec", "Non-Safety (BPCS)", "Pneumatic Diaphragm"],
        ["PCV-201", "LPG Sphere Overhead Depressurizing", "6 inch Class 300", "Fail Open (FO)", "< 4.0 sec", "SIL-2", "Pneumatic Piston"],
    ]
    for row in valves:
        ws2.append(row)

    wb.save(str(path))
    print(f"Created: {path.name}")


def create_hybrid_mah_safety_pdf():
    path = DATA_DIR / "09_Inspection_Accident" / "MRPL_MAH_Factory_Annual_Safety_Audit_2026.pdf"
    doc = pymupdf.open()

    # Page 1: Native digital text
    page1 = doc.new_page(width=612, height=792)
    p1_text = (
        "MRPL Mangalore Refinery and Petrochemicals Limited\n"
        "Major Accident Hazard (MAH) Factory Annual Safety Audit Report — 2026\n"
        "Audit Date: 14-February-2026 | Document Ref: MRPL-MAH-AUDIT-2026-REV3\n"
        "Audited Facilities: Crude Distillation (CDU-1/2), FCCU, Delayed Coker (DCU), LPG Tank Farm\n\n"
        "1. Executive Summary & Regulatory Authority\n"
        "This annual statutory safety audit was conducted in compliance with Rule 68-J of the Karnataka Factories Rules, 1969, "
        "and the Manufacture, Storage and Import of Hazardous Chemicals (MSIHC) Rules, 1989. "
        "The objective is to verify mechanical integrity, safety instrumented interlocks, fire protection readiness, "
        "and emergency response capability across all Major Accident Hazard units.\n\n"
        "2. General Audit Observations\n"
        "The refinery demonstrated robust process safety management practices across primary crude units. "
        "However, specific non-conformances were identified during physical field walkthroughs and instrumentation trip checks, "
        "as detailed in the attached inspection checklist on Page 2."
    )
    page1.insert_text((50, 60), p1_text, fontsize=11, fontname="helv")

    # Page 2: Scanned document simulation (Image rendered into PDF page without native text)
    img = Image.new("RGB", (1200, 1600), color="#FDFDFD")
    draw = ImageDraw.Draw(img)

    # Draw header and table lines
    draw.text((60, 60), "MRPL MAH STATUTORY SAFETY AUDIT — FIELD INSPECTION NON-CONFORMANCES", fill="#111111")
    draw.text((60, 95), "Location: Mangalore Refinery | Page 2 of 2 | Scanned Field Record", fill="#444444")
    draw.line([(50, 130), (1150, 130)], fill="#222222", width=3)

    findings = [
        "FINDING 01: Pressure transmitter PT-101 on FCCU Air Blower discharge exhibited heavy atmospheric salt corrosion on flange bolts.",
        "Corrective Action: Replace carbon steel studs with Inconel 625 bolting before next quarterly inspection. Target: 30 days.",
        "",
        "FINDING 02: Emergency Shutdown Valve XV-301 hydraulic actuator recorded a closing stroke time of 5.8 seconds during annual proof test.",
        "Non-Conformance: Exceeds mandatory SIL-3 threshold of 3.0 seconds maximum. Solenoid valve dump port partially restricted by particulate residue.",
        "Immediate Action: Flush hydraulic fluid, replace 24 VDC dual solenoid pilot assembly, and re-test within 48 hours.",
        "",
        "FINDING 03: Hydrocarbon gas detector GS-402 in LPG pump house showed zero drift exceeding 15% LEL calibration limit.",
        "Corrective Action: Sensor recalibrated with 50% LEL methane test gas; replaced sintered flame arrestor element.",
        "",
        "FINDING 04: Furnace F-101 pass flow meter FT-204 transmitter terminal block showed moisture ingress due to defective cable gland.",
        "Corrective Action: Resealed cable entry with flameproof compound per OISD-156 refinery electrical guidelines.",
    ]

    y = 160
    for line in findings:
        draw.text((60, y), line, fill="#111111")
        y += 45

    draw.line([(50, y + 40), (1150, y + 40)], fill="#222222", width=2)
    draw.text((60, y + 60), "Audited By: Chief Inspector of Factories & Lead Process Safety Auditor", fill="#111111")
    draw.text((60, y + 95), "Signed & Sealed: Verified Field Copy — Certified Confidential", fill="#333333")

    img_tmp = BASE_DIR / "scratch" / "scanned_mah_page.png"
    img_tmp.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(img_tmp))

    page2 = doc.new_page(width=612, height=792)
    page2.insert_image(pymupdf.Rect(20, 20, 592, 772), filename=str(img_tmp))

    doc.save(str(path))
    doc.close()
    print(f"Created hybrid PDF: {path.name}")


if __name__ == "__main__":
    create_fccu_manual()
    create_cdu_sop()
    create_oisd_standard()
    create_turnaround_sop()
    create_instrumentation_excel()
    create_hybrid_mah_safety_pdf()
    print("All MRPL refinery industrial reference documents generated successfully!")
