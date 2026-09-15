import os
from pathlib import Path
from typing import Dict, Any, List

def create_excel_report(data: List[Dict[str, Any]], filename: str) -> str:
    """Writes a list of dictionaries to an Excel file using pandas."""
    try:
        import pandas as pd
    except ImportError:
        return "Error: pandas is not installed. Run `pip install pandas`."
        
    try:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / filename
        
        df = pd.DataFrame(data)
        df.to_excel(filepath, index=False)
        return f"Successfully created Excel report at {filepath.absolute()}"
    except Exception as e:
        return f"Error creating Excel report: {e}"

def create_word_document(title: str, content: str, filename: str) -> str:
    """Generates a Word document using python-docx."""
    try:
        from docx import Document
    except ImportError:
        return "Error: python-docx is not installed. Run `pip install python-docx`."
        
    try:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / filename
        
        doc = Document()
        doc.add_heading(title, level=1)
        doc.add_paragraph(content)
        doc.save(filepath)
        return f"Successfully created Word document at {filepath.absolute()}"
    except Exception as e:
        return f"Error creating Word document: {e}"

def create_ppt_presentation(slides_data: List[Dict[str, str]], filename: str) -> str:
    """Generates a PowerPoint presentation using python-pptx."""
    try:
        from pptx import Presentation
    except ImportError:
        return "Error: python-pptx is not installed. Run `pip install python-pptx`."
        
    try:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / filename
        
        prs = Presentation()
        for slide_info in slides_data:
            title = slide_info.get("title", "")
            content = slide_info.get("content", "")
            
            # Use blank layout with title
            slide_layout = prs.slide_layouts[1] 
            slide = prs.slides.add_slide(slide_layout)
            
            title_shape = slide.shapes.title
            body_shape = slide.placeholders[1]
            
            title_shape.text = title
            tf = body_shape.text_frame
            tf.text = content
            
        prs.save(filepath)
        return f"Successfully created PPT presentation at {filepath.absolute()}"
    except Exception as e:
        return f"Error creating PPT presentation: {e}"
