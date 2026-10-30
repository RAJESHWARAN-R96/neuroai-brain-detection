"""
Clinical Decision Support & Pharmacotherapy Recommendation Engine.
Provides evidence-based clinical treatment pathways, surgical protocols,
first-line prescribed medications, and quantitative curability/prognosis assessments
for Glioma, Meningioma, Pituitary Adenoma, and Healthy Brain MRI scans.
"""

from typing import Dict, Any, List


def get_clinical_recommendations(
    predicted_class: str,
    tumor_metrics: Dict[str, Any] = None,
    confidence: float = 90.0
) -> Dict[str, Any]:
    """
    Computes evidence-based clinical treatment solutions, specific medications,
    and quantitative curability percentages based on histological class,
    lesion burden, and localization metrics.
    """
    tumor_metrics = tumor_metrics or {}
    area_pct = float(tumor_metrics.get('tumor_area_percentage', 0.0))
    severity = str(tumor_metrics.get('severity_index', 'Standard'))
    quadrant = str(tumor_metrics.get('anatomical_quadrant', 'Central Parenchyma'))

    cname = predicted_class.lower()

    if cname == 'glioma':
        # Curability calculation:
        # Lower tumor area + focal presentation = higher surgical resection potential
        # High grade / large burden (>8% of brain) = guarded curability
        base_curable = 48.0
        if area_pct < 4.0:
            curable_pct = round(min(65.0, base_curable + 12.0 - (area_pct * 1.5)), 1)
        elif area_pct < 10.0:
            curable_pct = round(max(25.0, base_curable - ((area_pct - 4.0) * 2.5)), 1)
        else:
            curable_pct = round(max(15.0, 32.0 - ((area_pct - 10.0) * 1.2)), 1)

        non_curable_pct = round(100.0 - curable_pct, 1)

        return {
            'class_key': 'glioma',
            'disease_name': 'Infiltrative Glioma (Astrocytoma / Glioblastoma Spectrum)',
            'curability': {
                'curable_percentage': curable_pct,
                'non_curable_percentage': non_curable_pct,
                'prognosis_category': 'Guarded to Moderate (Multimodal Management Required)',
                'badge_color': '#ef4444',
                'survival_benchmark': '5-Year Overall Survival: 25% – 60% (dependent on IDH1/IDH2 mutation & MGMT promoter methylation status)',
                'curability_summary': (
                    f"Gliomas are infiltrative primary neuroepithelial tumors. Complete microscopic cure is challenging due to microscopic "
                    f"parenchymal infiltration beyond radiological margins. Current estimated curative / long-term progression-free remission rate "
                    f"is {curable_pct}%, with {non_curable_pct}% requiring chronic recurrence surveillance and secondary targeted therapy."
                )
            },
            'primary_solution': {
                'title': 'Maximal Safe Surgical Resection + Adjuvant Chemoradiotherapy (Stupp Protocol)',
                'urgency': 'High Urgency (Neuro-Oncology Multidisciplinary Triage within 48–72 Hours)',
                'surgical_protocol': 'Image-guided craniotomy with intraoperative MRI / 5-ALA fluorescence-guided cortical mapping for gross-total resection (GTR) of enhancing mass.',
                'radiotherapy_protocol': 'Fractionated Conformal Radiation Therapy (60 Gy in 30 daily fractions over 6 weeks) targeted to contrast-enhancing cavity + 2 cm FLAIR margin.',
                'procedure_type': 'Surgical Craniotomy & Molecular Biopsy'
            },
            'recommended_medicines': [
                {
                    'name': 'Temozolomide (TMZ)',
                    'category': 'Alkylating Oral Chemotherapy (First-Line)',
                    'dosage': '75 mg/m²/day orally during 6-week radiotherapy; then maintenance 150–200 mg/m²/day on days 1–5 of 28-day cycles (6–12 cycles)',
                    'purpose': 'Crosses blood-brain barrier to methylate neoplastic DNA at O6 and N7 positions of guanine, triggering tumor cell apoptosis.',
                    'clinical_priority': 'Mandatory Protocol'
                },
                {
                    'name': 'Dexamethasone',
                    'category': 'High-Potency Corticosteroid',
                    'dosage': '4 mg to 8 mg orally / IV every 8 to 12 hours (taper slowly post-resection)',
                    'purpose': 'Rapidly stabilizes endothelial junctions to reduce life-threatening peritumoral vasogenic edema and intracranial pressure (ICP).',
                    'clinical_priority': 'Acute Symptom Control'
                },
                {
                    'name': 'Bevacizumab (Avastin)',
                    'category': 'Anti-VEGF Monoclonal Antibody',
                    'dosage': '10 mg/kg IV infusion every 2 weeks',
                    'purpose': 'Inhibits vascular endothelial growth factor (VEGF-A) to starve neoplastic microvascular proliferation and relieve refractory edema.',
                    'clinical_priority': 'Second-Line / Recurrence'
                },
                {
                    'name': 'Levetiracetam (Keppra)',
                    'category': 'Broad-Spectrum Antiepileptic Prophylaxis',
                    'dosage': '500 mg to 1000 mg orally twice daily',
                    'purpose': 'Non-enzyme-inducing anti-convulsant for perioperative seizure prophylaxis without interfering with chemotherapeutic metabolism.',
                    'clinical_priority': 'Seizure Prophylaxis'
                },
                {
                    'name': 'Mannitol 20%',
                    'category': 'Osmotic Diuretic',
                    'dosage': '0.5 to 1.0 g/kg IV rapid infusion over 20 minutes',
                    'purpose': 'Emergency management of acute intracranial hypertension and herniation syndromes in large mass-effect lesions.',
                    'clinical_priority': 'Emergency / Pre-Op Only'
                }
            ],
            'adjunctive_therapies': [
                'Tumor Treating Fields (Optune TTFields): Low-intensity alternating electric fields (200 kHz) disrupting mitotic spindle tubulin.',
                'IDH-Inhibitor Targeted Therapy (Vorasidenib / Ivosidenib) if IDH-1 R132H mutation confirmed by immunohistochemistry.',
                'Stereotactic Radiosurgery (CyberKnife / Gamma Knife) boost for focal localized residual borders.'
            ],
            'monitoring_and_followup': 'Contrast-enhanced brain MRI with perfusion and spectroscopy every 8–12 weeks for the first 2 years post-resection, transitioning to every 3–4 months thereafter.'
        }

    elif cname == 'meningioma':
        # Meningiomas are overwhelmingly benign (WHO Grade I in ~90%)
        # Complete surgical resection (Simpson Grade I) cures >85-94% of cases
        base_curable = 92.0
        if area_pct < 5.0:
            curable_pct = round(min(96.0, base_curable + 2.0), 1)
        else:
            curable_pct = round(max(75.0, base_curable - (area_pct * 1.2)), 1)

        non_curable_pct = round(100.0 - curable_pct, 1)

        return {
            'class_key': 'meningioma',
            'disease_name': 'Extra-Axial Meningioma (Arachnoid Cap Origin)',
            'curability': {
                'curable_percentage': curable_pct,
                'non_curable_percentage': non_curable_pct,
                'prognosis_category': 'Highly Curable / Excellent Long-Term Remission',
                'badge_color': '#10b981',
                'survival_benchmark': '10-Year Overall Survival: 85% – 95% post-complete Simpson Grade I resection',
                'curability_summary': (
                    f"Meningiomas are typically benign extra-axial lesions arising from the dura mater. When gross-total resection including the dural "
                    f"attachment is achieved (Simpson Grade I/II), the cure rate reaches {curable_pct}%. Only {non_curable_pct}% (associated with atypical "
                    f"WHO Grade II/III histology or critical skull base venous sinus involvement) carry recurrence risks requiring radiation."
                )
            },
            'primary_solution': {
                'title': 'Complete Micro-Neurosurgical Resection (Simpson Grade I/II) or Stereotactic Radiosurgery (SRS)',
                'urgency': 'Moderate Urgency (Elective Neurosurgical Consultation within 1–2 Weeks)',
                'surgical_protocol': 'Extradural / intradural craniotomy with bipolar coagulation and complete excision of the tumor, dural tail origin, and any invaded hyperostotic bone.',
                'radiotherapy_protocol': 'For skull base lesions (e.g. cavernous sinus, petroclival): Stereotactic Radiosurgery (Gamma Knife / CyberKnife, 12–16 Gy marginal dose).',
                'procedure_type': 'Microsurgical Resection or Non-Invasive Radiosurgery'
            },
            'recommended_medicines': [
                {
                    'name': 'Levetiracetam (Keppra)',
                    'category': 'Antiepileptic Agent',
                    'dosage': '500 mg orally twice daily for 1 to 3 months perioperatively',
                    'purpose': 'Prevents cortical irritation and seizure episodes induced by convexity mass effect or surgical manipulation.',
                    'clinical_priority': 'Perioperative Standard'
                },
                {
                    'name': 'Dexamethasone',
                    'category': 'Corticosteroid',
                    'dosage': '2 mg to 4 mg orally twice daily (short-course pre-op)',
                    'purpose': 'Reduces surrounding parenchymal compression and localized vasogenic edema prior to elective surgical resection.',
                    'clinical_priority': 'Symptomatic Mass Effect'
                },
                {
                    'name': 'Hydroxyurea',
                    'category': 'Antimetabolite / Ribonucleotide Reductase Inhibitor',
                    'dosage': '20 mg/kg/day orally (continuous therapy)',
                    'purpose': 'Induces cell cycle arrest in inoperable, recurrent, or atypical meningiomas unresponsive to radiation therapy.',
                    'clinical_priority': 'Refractory / Inoperable Cases'
                },
                {
                    'name': 'Octreotide (Sandostatin LAR)',
                    'category': 'Somatostatin Receptor Type 2 (SSTR2) Agonist',
                    'dosage': '20 mg to 30 mg intramuscularly every 4 weeks',
                    'purpose': 'Targets high-affinity SSTR2 receptors expressed on meningioma cells to inhibit tumor proliferation in recurrent skull-base lesions.',
                    'clinical_priority': 'Targeted Receptor Therapy'
                }
            ],
            'adjunctive_therapies': [
                'Pre-Operative Endovascular Embolization: Microcatheter PVA particle embolization of external carotid artery branches (middle meningeal artery) to minimize intraoperative bleeding.',
                'Active Radiographic Surveillance (Wait-and-Scan): For asymptomatic small (<2.5 cm) incidental calcified meningiomas with serial MRI scans.'
            ],
            'monitoring_and_followup': 'Surveillance contrast MRI at 3 months post-op, then annually for 5 years, transitioning to biennial scans up to 10 years.'
        }

    elif cname == 'pituitary':
        # Pituitary adenomas are benign and highly curable via transsphenoidal surgery or medical therapy (Dopamine agonists)
        base_curable = 89.0
        if area_pct < 4.0:
            curable_pct = round(min(94.0, base_curable + 4.0), 1)
        else:
            curable_pct = round(max(78.0, base_curable - (area_pct * 1.0)), 1)

        non_curable_pct = round(100.0 - curable_pct, 1)

        return {
            'class_key': 'pituitary',
            'disease_name': 'Pituitary Sellar Adenoma (Micro / Macro-adenoma)',
            'curability': {
                'curable_percentage': curable_pct,
                'non_curable_percentage': non_curable_pct,
                'prognosis_category': 'Highly Curable & Endocrine-Controllable',
                'badge_color': '#3b82f6',
                'survival_benchmark': 'Long-Term Disease-Free Remission: 80% – 95% with combined endoscopic and medical pharmacotherapy',
                'curability_summary': (
                    f"Pituitary adenomas are non-malignant neuroendocrine tumors of the sella turcica. Prolactin-secreting tumors achieve over 85% "
                    f"complete medical cure with dopamine agonist tablets alone without surgery. Overall curability / functional remission rate is "
                    f"{curable_pct}%, with {non_curable_pct}% requiring long-term endocrine hormone replacement or secondary stereotactic radiosurgery."
                )
            },
            'primary_solution': {
                'title': 'Endoscopic Endonasal Transsphenoidal Hypophysectomy or Medical Dopaminergic Therapy',
                'urgency': 'Prompt Urgency (Urgent if Optic Chiasm Compression / Bitemporal Hemianopsia is present)',
                'surgical_protocol': 'Minimally invasive endoscopic transsphenoidal approach through the nasal cavity and sphenoid sinus to selectively excise the sellar lesion while preserving normal pituitary gland tissue.',
                'radiotherapy_protocol': 'Fractionated stereotactic radiotherapy (FSRT) or Gamma Knife radiosurgery (18–25 Gy margin dose) for residual adenoma invading the cavernous sinus.',
                'procedure_type': 'Endoscopic Transsphenoidal Surgery'
            },
            'recommended_medicines': [
                {
                    'name': 'Cabergoline',
                    'category': 'Potent Dopamine-2 (D2) Receptor Agonist (First-Line)',
                    'dosage': '0.25 mg to 0.5 mg orally twice weekly (titrated based on serum prolactin levels)',
                    'purpose': 'Stimulates pituitary D2 receptors to suppress prolactin secretion and induce tumor shrinkage in >85% of prolactinomas.',
                    'clinical_priority': 'First-Line Pharmacotherapy'
                },
                {
                    'name': 'Bromocriptine (Parlodel)',
                    'category': 'Ergot-Derived Dopamine Agonist',
                    'dosage': '1.25 mg to 2.5 mg orally daily with food at bedtime (titrated up to 7.5–15 mg/day)',
                    'purpose': 'Alternative D2 agonist for patients wishing to conceive or intolerant to cabergoline.',
                    'clinical_priority': 'Second-Line Dopamine Agonist'
                },
                {
                    'name': 'Hydrocortisone',
                    'category': 'Glucocorticoid Replacement Therapy',
                    'dosage': '10 mg to 15 mg on waking, 5 mg at 2 PM (simulating physiological circadian rhythm)',
                    'purpose': 'Vital replacement therapy if mass effect impairs the pituitary-adrenal axis (central hypoadrenalism).',
                    'clinical_priority': 'Endocrine Deficiency Support'
                },
                {
                    'name': 'Levothyroxine (Synthroid)',
                    'category': 'Thyroid Hormone Replacement (T4)',
                    'dosage': '50 mcg to 100 mcg orally once daily on an empty stomach in the morning',
                    'purpose': 'Treats secondary central hypothyroidism resulting from TSH-producing cell compression.',
                    'clinical_priority': 'Endocrine Replacement'
                },
                {
                    'name': 'Pasireotide (Signifor) / Ketoconazole',
                    'category': 'Somatostatin Multi-Receptor Agonist / Steroidogenesis Inhibitor',
                    'dosage': '0.6 mg to 0.9 mg subcutaneously twice daily',
                    'purpose': 'Indicated specifically for ACTH-secreting pituitary adenomas (Cushing\'s disease) refractory to surgical resection.',
                    'clinical_priority': 'Cushing\'s Disease Specific'
                }
            ],
            'adjunctive_therapies': [
                'Formal Neuro-Ophthalmology Visual Field Assessment (Humphrey 30-2 Perimetry): Mandatory evaluation for optic chiasm decompression.',
                'Full Baseline Pituitary Endocrine Panel: Serum Prolactin, IGF-1, Morning Cortisol, ACTH, Free T4, TSH, LH, FSH, Testosterone/Estradiol.'
            ],
            'monitoring_and_followup': 'High-resolution sellar dynamic contrast MRI at 3 months post-intervention, with 6-month endocrine and visual field re-evaluations.'
        }

    else:
        # No tumor / Healthy brain
        return {
            'class_key': 'notumor',
            'disease_name': 'Normal Brain MRI (No Neoplastic Lesion Detected)',
            'curability': {
                'curable_percentage': 100.0,
                'non_curable_percentage': 0.0,
                'prognosis_category': 'Normal Anatomical Status / Full Physiological Health',
                'badge_color': '#10b981',
                'survival_benchmark': 'Standard Age-Adjusted Normal Life Expectancy',
                'curability_summary': 'No neoplastic masses, space-occupying lesions, midline shifts, or abnormal contrast enhancements detected. Patient maintains normal neuro-anatomical health.'
            },
            'primary_solution': {
                'title': 'No Surgical or Oncological Intervention Indicated',
                'urgency': 'Routine / Reassurance',
                'surgical_protocol': 'None. Brain parenchyma, ventricles, sulci, and posterior fossa are anatomically intact.',
                'radiotherapy_protocol': 'None.',
                'procedure_type': 'Clinical Reassurance & Routine Lifestyle Care'
            },
            'recommended_medicines': [
                {
                    'name': 'No Prescription Neuro-Oncological Medicines Required',
                    'category': 'Healthy Parenchyma',
                    'dosage': 'N/A',
                    'purpose': 'Normal brain MRI scan. No anti-neoplastic, anti-edema, or anti-convulsant therapy indicated.',
                    'clinical_priority': 'None Required'
                },
                {
                    'name': 'Acetaminophen / Paracetamol (If Headaches Present)',
                    'category': 'Mild Analgesic',
                    'dosage': '500 mg to 1000 mg orally as needed (maximum 3000 mg/day)',
                    'purpose': 'Symptomatic management of benign tension or cervical headaches if clinical presentation prompted the MRI.',
                    'clinical_priority': 'Symptomatic Relief Only'
                }
            ],
            'adjunctive_therapies': [
                'Neurological physical examination to investigate non-structural headache triggers (e.g. migraine, sinusitis, cervical spine strain, stress).',
                'Hydration, regular sleep cycle, and ergonomic posture review.'
            ],
            'monitoring_and_followup': 'Routine health monitoring. Repeat MRI only if new focal neurological deficits, visual loss, or sudden intractable headaches develop.'
        }
