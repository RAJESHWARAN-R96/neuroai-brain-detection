"""
Clinical PDF Diagnostic Report Generator using ReportLab.
Generates comprehensive diagnostic reports with embedded MRI scans, Grad-CAM heatmaps,
class probability breakdowns, tumor localization metrics, and clinician signature blocks.
"""

import os
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from .dataset import CLASS_DESCRIPTIONS


def generate_pdf_report(
    patient_info: Dict[str, Any],
    diagnosis_results: Dict[str, Any],
    original_img_path: str,
    preprocessed_img_path: str,
    gradcam_img_path: str,
    output_pdf_path: str
) -> str:
    """
    Renders a multi-section clinical PDF report and saves it to output_pdf_path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        alignment=0
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b')
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    
    body_bold = ParagraphStyle(
        'BodyDarkBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    elements = []
    
    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>NEURO-AI DIAGNOSTIC CLINICAL REPORT</b>", title_style),
            Paragraph("<b>METAHEURISTIC OPTIMIZED AI</b><br/>Confidential Medical Document", subtitle_style)
        ]
    ]
    t_header = Table(header_data, colWidths=[360, 180])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
    ]))
    elements.append(t_header)
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0284c7'), spaceBefore=4, spaceAfter=8))
    
    # 2. Patient & Scan Metadata Table
    patient_id = patient_info.get('patient_id', f"PAT-{int(time.time()) % 100000}")
    patient_name = patient_info.get('patient_name', 'Anonymous Patient')
    scan_date = patient_info.get('scan_date', time.strftime("%Y-%m-%d %H:%M:%S"))
    ai_model = diagnosis_results.get('model_used', 'GWO-Optimized Deep Vision Ensemble')
    
    meta_table_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {patient_id}", body_style),
            Paragraph(f"<b>Patient Name:</b> {patient_name}", body_style),
            Paragraph(f"<b>Scan Date:</b> {scan_date}", body_style)
        ],
        [
            Paragraph("<b>Modality:</b> Brain MRI (T1-CE / Axial)", body_style),
            Paragraph("<b>AI Engine:</b> Metaheuristic Multi-CNN", body_style),
            Paragraph(f"<b>Architecture:</b> {ai_model}", body_style)
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[180, 180, 180])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#f1f5f9')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    elements.append(t_meta)
    elements.append(Spacer(1, 10))
    
    # 3. Diagnostic Verdict Summary Card
    pred_class = diagnosis_results.get('predicted_class', 'notumor')
    confidence = diagnosis_results.get('confidence', 95.0)
    class_info = CLASS_DESCRIPTIONS.get(pred_class, CLASS_DESCRIPTIONS['notumor'])
    
    if pred_class == 'notumor':
        badge_color = colors.HexColor('#059669') # Green
        badge_bg = colors.HexColor('#ecfdf5')
        verdict_text = f"NORMAL SCAN — NO TUMOR DETECTED (Confidence: {confidence:.1f}%)"
    else:
        badge_color = colors.HexColor('#dc2626') # Red
        badge_bg = colors.HexColor('#fef2f2')
        verdict_text = f"PATHOLOGICAL FINDING: {class_info['title'].upper()} DETECTED (Confidence: {confidence:.1f}%)"
        
    verdict_style = ParagraphStyle(
        'Verdict',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=badge_color,
        alignment=1
    )
    
    t_verdict = Table([[Paragraph(verdict_text, verdict_style)]], colWidths=[540])
    t_verdict.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), badge_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, badge_color),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    ]))
    elements.append(t_verdict)
    elements.append(Spacer(1, 8))
    
    # 4. Multi-View Imaging Panel (Original, Preprocessed, Grad-CAM Heatmap)
    elements.append(Paragraph("<b>MULTI-STAGE MRI VISUALIZATION & EXPLAINABILITY (XAI)</b>", section_heading))
    
    img_w, img_h = 165, 165
    
    # Verify image paths
    rl_orig = RLImage(original_img_path, width=img_w, height=img_h) if os.path.isfile(original_img_path) else Paragraph("N/A", body_style)
    rl_prep = RLImage(preprocessed_img_path, width=img_w, height=img_h) if os.path.isfile(preprocessed_img_path) else Paragraph("N/A", body_style)
    rl_xai = RLImage(gradcam_img_path, width=img_w, height=img_h) if os.path.isfile(gradcam_img_path) else Paragraph("N/A", body_style)
    
    images_table_data = [
        [rl_orig, rl_prep, rl_xai],
        [
            Paragraph("<b>1. Raw MRI Acquisition</b><br/><font color='#64748b'>Unprocessed input scan</font>", body_style),
            Paragraph("<b>2. Skull-Stripped & CLAHE</b><br/><font color='#64748b'>Contour cropped & denoised</font>", body_style),
            Paragraph("<b>3. Grad-CAM++ Saliency</b><br/><font color='#64748b'>AI Attention Tumor Map</font>", body_style)
        ]
    ]
    t_imgs = Table(images_table_data, colWidths=[180, 180, 180])
    t_imgs.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 4)
    ]))
    elements.append(t_imgs)
    elements.append(Spacer(1, 10))
    
    # 5. Probability Breakdown & Lesion Metrics
    elements.append(Paragraph("<b>DIAGNOSTIC PROBABILITY DISTRIBUTION & TUMOR LOCALIZATION</b>", section_heading))
    
    probs = diagnosis_results.get('probabilities', {'glioma': 0, 'meningioma': 0, 'notumor': 0, 'pituitary': 0})
    tumor_metrics = diagnosis_results.get('tumor_metrics', {})
    
    prob_rows = [
        [Paragraph("<b>Pathology / Class</b>", body_bold), Paragraph("<b>Probability</b>", body_bold), Paragraph("<b>Clinical Risk</b>", body_bold)]
    ]
    for cname in ['glioma', 'meningioma', 'pituitary', 'notumor']:
        p_val = probs.get(cname, 0.0)
        c_meta = CLASS_DESCRIPTIONS[cname]
        prob_rows.append([
            Paragraph(c_meta['title'], body_style),
            Paragraph(f"<b>{p_val:.1f}%</b>", body_style),
            Paragraph(f"<font color='{c_meta['color']}'>{c_meta['urgency']}</font>", body_style)
        ])
        
    t_prob = Table(prob_rows, colWidths=[110, 80, 80])
    t_prob.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    
    # Tumor localization table
    area_pct = tumor_metrics.get('tumor_area_percentage', 0.0)
    quadrant = tumor_metrics.get('anatomical_quadrant', 'Central Parenchyma')
    severity = tumor_metrics.get('severity_index', 'Standard')
    
    metric_rows = [
        [Paragraph("<b>Quantitative Lesion Metric</b>", body_bold), Paragraph("<b>Observation Value</b>", body_bold)],
        [Paragraph("Estimated Tumor Area Burden", body_style), Paragraph(f"<b>{area_pct:.2f}%</b> of brain tissue", body_style)],
        [Paragraph("Anatomical Localization", body_style), Paragraph(f"<b>{quadrant}</b>", body_style)],
        [Paragraph("Mass Effect Index", body_style), Paragraph(f"<b>{severity}</b>", body_style)],
        [Paragraph("Pathological Description", body_style), Paragraph(f"{class_info['description']}", body_style)]
    ]
    t_metrics = Table(metric_rows, colWidths=[130, 140])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))
    
    # Side-by-side tables
    t_side = Table([[t_prob, t_metrics]], colWidths=[270, 270])
    t_side.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('PADDING', (0, 0), (-1, -1), 0)
    ]))
    elements.append(t_side)
    elements.append(Spacer(1, 10))
    
    # 6. Clinical Guidance, Curability & Recommended Medications
    clinical_guidance = diagnosis_results.get('clinical_guidance')
    if clinical_guidance:
        curable_info = clinical_guidance.get('curability', {})
        sol_info = clinical_guidance.get('primary_solution', {})
        medicines = clinical_guidance.get('recommended_medicines', [])
        
        elements.append(Paragraph("<b>CLINICAL PROGNOSIS, CURABILITY & PHARMACOTHERAPY RECOMMENDATIONS</b>", section_heading))
        
        curable_pct = curable_info.get('curable_percentage', 0.0)
        non_curable_pct = curable_info.get('non_curable_percentage', 0.0)
        prog_cat = curable_info.get('prognosis_category', 'Clinical Assessment')
        
        # Curability Banner Table
        curable_text = f"<b>ESTIMATED CURABILITY RATE: {curable_pct}%</b> &nbsp;|&nbsp; Chronic / Recurrence Risk: {non_curable_pct}% &nbsp;|&nbsp; <b>Prognosis:</b> {prog_cat}"
        t_curable = Table([[Paragraph(curable_text, body_style)]], colWidths=[540])
        t_curable.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ecfdf5') if curable_pct >= 80 else (colors.HexColor('#eff6ff') if curable_pct >= 60 else colors.HexColor('#fef2f2'))),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#10b981') if curable_pct >= 80 else (colors.HexColor('#3b82f6') if curable_pct >= 60 else colors.HexColor('#ef4444'))),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]))
        elements.append(t_curable)
        elements.append(Spacer(1, 6))
        
        # Treatment Solution Summary
        sol_text = f"<b>Recommended Primary Solution:</b> {sol_info.get('title', 'N/A')}<br/><b>Surgical / Clinical Pathway:</b> {sol_info.get('surgical_protocol', 'N/A')}"
        elements.append(Paragraph(sol_text, body_style))
        elements.append(Spacer(1, 6))
        
        # Recommended Medications Table
        if medicines:
            med_rows = [
                [Paragraph("<b>Recommended Medicine</b>", body_bold), Paragraph("<b>Drug Category</b>", body_bold), Paragraph("<b>Clinical Dosage & Purpose</b>", body_bold)]
            ]
            for med in medicines[:4]:  # Top 4 medications for neat PDF formatting
                med_rows.append([
                    Paragraph(f"<b>{med.get('name', 'N/A')}</b>", body_style),
                    Paragraph(med.get('category', 'N/A'), body_style),
                    Paragraph(f"<b>Dosage:</b> {med.get('dosage', 'N/A')}<br/><font color='#64748b'>{med.get('purpose', '')}</font>", body_style)
                ])
                
            t_meds = Table(med_rows, colWidths=[120, 130, 290])
            t_meds.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8fafc')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('PADDING', (0, 0), (-1, -1), 4),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            elements.append(t_meds)
            elements.append(Spacer(1, 8))
    
    # 7. Clinician Sign-off and Disclaimer
    elements.append(KeepTogether([
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=4, spaceAfter=8),
        Table([
            [
                Paragraph("<b>Attending Radiologist / Clinician Sign-Off:</b><br/><br/>________________________________________<br/>Signature & Stamp", body_style),
                Paragraph("<b>AI Assistance Disclaimer:</b><br/><i>This report is generated by a deep learning metaheuristic framework as a clinical decision support tool. Final clinical diagnosis must be verified by a board-certified radiologist.</i>", subtitle_style)
            ]
        ], colWidths=[270, 270])
    ]))
    
    doc.build(elements)
    return output_pdf_path
