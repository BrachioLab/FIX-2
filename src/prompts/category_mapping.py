category_mapping_massmaps = {
    'name2id': {
        'Lensing Peak (Cluster) Abundance': 1,
        'Void Size and Frequency': 2,
        'Filament Thickness and Sharpness': 3,
        'Fine-Scale Clumpiness': 4,
        'Connectivity of the Cosmic Web': 5,
        'Density Contrast Extremes': 6,
    },
    'id2name': {
        1: 'Lensing Peak (Cluster) Abundance',
        2: 'Void Size and Frequency',
        3: 'Filament Thickness and Sharpness',
        4: 'Fine-Scale Clumpiness',
        5: 'Connectivity of the Cosmic Web',
        6: 'Density Contrast Extremes',
    },
    'name2description': {
        'Lensing Peak (Cluster) Abundance': 'A higher count of prominent, high-convergence peaks in the map indicates a larger sigma_8, since a clumpier matter distribution produces more frequent massive halos.',
        'Void Size and Frequency': 'Extensive low-convergence void regions suggest a lower Omega_m, as a reduced overall matter density allows bigger underdense expanses to form in the cosmic web.',
        'Filament Thickness and Sharpness': 'Bold, sharply defined filaments threading between clusters imply a higher sigma_8 (stronger small-scale clustering), whereas thin or diffuse filaments point to a lower amplitude of matter fluctuations.',
        'Fine-Scale Clumpiness': 'A grainy, fine-textured pattern of small-scale lensing fluctuations (many mini-clumps) is a visual signature of high sigma_8, whereas a smoother, more homogeneous map suggests a lower sigma_8.',
        'Connectivity of the Cosmic Web': 'A highly interconnected filament network (with filaments linking most clusters into a continuous web) hints at a higher Omega_m, whereas a more fragmented scene of isolated clumps separated by wide gaps is expected for a lower Omega_m.',
        'Density Contrast Extremes': 'Very pronounced contrast between dense regions and empty voids - i.e. bright lensing peaks adjacent to dark void areas - signals an enhanced variance of the density field (high sigma_8), whereas subdued contrast suggests lower sigma_8.',
    }
}


category_mapping_cholec = {
    'name2id': {
        "Calot's triangle cleared": 1,
        'Cystic plate exposed': 2,
        'Only two structures visible': 3,
        'Above the R4U line': 4,
        'Safe distance from common bile duct': 5,
        'Infundibulum start point': 6,
        'Subserosal plane stay': 7,
        'Cystic lymph node guide': 8,
        'No division without ID': 9,
        'Inflammation bailout': 10,
        'Aberrant artery caution': 11,
    },
    'id2name': {
        1: "Calot's triangle cleared",
        2: 'Cystic plate exposed',
        3: 'Only two structures visible',
        4: 'Above the R4U line',
        5: 'Safe distance from common bile duct',
        6: 'Infundibulum start point',
        7: 'Subserosal plane stay',
        8: 'Cystic lymph node guide',
        9: 'No division without ID',
        10: 'Inflammation bailout',
        11: 'Aberrant artery caution',
    },
    'name2description': {
        "Calot's triangle cleared": "Hepatocystic triangle must be fully cleared of fat/fibrosis so that its boundaries are unmistakable.",
        "Cystic plate exposed": "The lower third of the gallbladder must be dissected off the liver to reveal the shiny cystic plate and ensure the correct dissection plane.",
        "Only two structures visible": "Only the cystic duct and cystic artery should be seen entering the gallbladder before any clipping or cutting.",
        "Above the R4U line": "Dissection must remain cephalad to an imaginary line from Rouviere's sulcus to liver segment IV umbilical fissure to avoid the common bile duct. Dissection should be carried out along the inferior edge of the gallbladder (well above the line of safety).",
        "Safe distance from common bile duct": "Dissection must maintain a safe distance from the common bile duct to prevent injury.",
        "Infundibulum start point": "Dissection can begin at the gallbladder infundibulum-cystic duct junction to stay in safe tissue planes, or at the lateral or medial edges of the gallbladder above Rouviere’s sulcus or along the cystic plate to get mobility of the gallbladder first.",
        "Subserosal plane stay": "When separating the gallbladder from the liver, stay in the avascular peritoneal cleavage plane.",
        "Cystic lymph node guide": "Identify the cystic lymph node and clip the artery on the gallbladder side of the node to avoid injuring the hepatic artery.",
        "No division without ID": "Never divide any duct or vessel until it is unequivocally identified as the cystic structure entering the gallbladder.",
        "Inflammation bailout": "If dense scarring or distorted anatomy obscures Calot's triangle, convert to open surgery or a fenestrated subtotal approach rather than blind cutting.",
        "Aberrant artery caution": "Exercise caution when aberrant arteries are present to avoid vascular injury.",
    }
}

category_mapping_supernova = {
    'name2id': {
        'Contiguous non-zero flux': 1,
        'Rise–decline rates': 2,
        'Photometric amplitude': 3,
        'Event duration': 4,
        'Periodic light curves': 5,
        'Secondary maxima': 6,
        'Monotonic flux trends': 7,
    },
    'id2name': {
        1: 'Contiguous non-zero flux',
        2: 'Rise–decline rates',
        3: 'Photometric amplitude',
        4: 'Event duration',
        5: 'Periodic light curves',
        6: 'Secondary maxima',
        7: 'Monotonic flux trends',
    },
    'name2description': {
        'Contiguous non-zero flux': 'Contiguous non‑zero flux segments confirm genuine astrophysical activity and define the time windows from which transient features should be extracted.',
        'Rise–decline rates': 'Characteristic rise‑and‑decline rates—such as the fast‑rise/slow‑fade morphology of many supernovae—encode energy‑release physics and serve as strong class discriminators.',
        'Photometric amplitude': 'Peak‑to‑trough photometric amplitude separates high‑energy explosive events (multi‑magnitude outbursts) from low‑amplitude periodic or stochastic variables.',
        'Event duration': 'Total event duration, measured from first detection to return to baseline, distinguishes short‑lived kilonovae and superluminous SNe from longer plateau or AGN variability phases.',
        'Periodic light curves': 'Periodic light curves with stable periods and distinctive Fourier amplitude‑ and phase‑ratios (e.g., φ21, φ31) flag pulsators and eclipsing binaries rather than one‑off transients.',
        'Secondary maxima': 'Filter‑specific secondary maxima or shoulders in red/near‑IR bands—prominent in SNe Ia—are morphological features absent in most core‑collapse SNe.',
        'Monotonic flux trends': 'Locally smooth, monotonic flux trends across one or multiple bands (plateaus, linear decays) capture physical evolution stages and help distinguish SN II‑P, SN II‑L, and related classes.',
    }
}

category_mapping_sepsis = {
    'name2id': {
        'Elderly Susceptibility (Age ≥65 years)': 1,
        'SIRS Positivity (≥2 Criteria)': 2,
        'High qSOFA Score (≥2)': 3,
        'Elevated NEWS Score (≥5 points)': 4,
        'Elevated Serum Lactate (≥2 mmol/L)': 5,
        'Elevated Shock Index (≥1.0)': 6,
        'Sepsis-Associated Hypotension (SBP <90 mmHg or MAP <70 mmHg, or ≥40 mmHg drop)': 7,
        'SOFA Score Increase (≥2 points)': 8,
        'Early Antibiotic/Culture Orders (within 2 hours)': 9,
    },
    'id2name': {
        1: 'Elderly Susceptibility (Age ≥65 years)',
        2: 'SIRS Positivity (≥2 Criteria)',
        3: 'High qSOFA Score (≥2)',
        4: 'Elevated NEWS Score (≥5 points)',
        5: 'Elevated Serum Lactate (≥2 mmol/L)',
        6: 'Elevated Shock Index (≥1.0)',
        7: 'Sepsis-Associated Hypotension (SBP <90 mmHg or MAP <70 mmHg, or ≥40 mmHg drop)',
        8: 'SOFA Score Increase (≥2 points)',
        9: 'Early Antibiotic/Culture Orders (within 2 hours)',
    },
    'name2description': {
        'Elderly Susceptibility (Age ≥65 years)': 'Advanced age (≥ 65 years) markedly increases susceptibility to rapid sepsis progression and higher mortality after infection.',
        'SIRS Positivity (≥2 Criteria)': 'Presence of ≥ 2 SIRS criteria—temperature > 38 °C or < 36 °C, heart rate > 90 bpm, respiratory rate > 20 /min or PaCO₂ < 32 mm Hg, or WBC > 12 000/µL or < 4 000/µL—identifies systemic inflammation consistent with early sepsis.',
        'High qSOFA Score (≥2)': 'A qSOFA score ≥ 2 (respiratory rate ≥ 22 /min, systolic BP ≤ 100 mmHg, or altered mentation) flags high risk of sepsis‑related organ dysfunction and mortality.',
        'Elevated NEWS Score (≥5 points)': 'A National Early Warning Score (NEWS) of ≥ 5–7 derived from deranged vitals predicts imminent clinical deterioration compatible with sepsis.',
        'Elevated Serum Lactate (≥2 mmol/L)': 'Serum lactate ≥ 2 mmol/L within the first 2 hours signals tissue hypoperfusion and markedly elevates sepsis mortality risk.',
        'Elevated Shock Index (≥1.0)': 'Shock index (heart rate ÷ systolic BP) ≥ 1.0—or a rise ≥ 0.3 from baseline—denotes haemodynamic instability and a high probability of severe sepsis.',
        'Sepsis-Associated Hypotension (SBP <90 mmHg or MAP <70 mmHg, or ≥40 mmHg drop)': 'Sepsis‑associated hypotension, defined as SBP < 90 mmHg, MAP < 70 mmHg, or a ≥ 40 mmHg drop from baseline, indicates progression toward septic shock.',
        'SOFA Score Increase (≥2 points)': 'An increase of ≥ 2 points in any SOFA component—e.g., PaO₂/FiO₂ < 300, platelets < 100 × 10⁹/L, bilirubin > 2 mg/dL, creatinine > 2 mg/dL, or GCS < 12—confirms new organ dysfunction and high sepsis risk.',
        'Early Antibiotic/Culture Orders (within 2 hours)': 'Administration of broad‑spectrum antibiotics or drawing of blood cultures within the first 2 hours signifies clinician suspicion of serious infection and should anchor sepsis risk assessment.',
    }
}

category_mapping_cardiac = {
    'name2id': {
        'Ventricular Tachyarrhythmias': 1,
        'Ventricular Ectopy / NSVT': 2,
        'Bradycardia or Heart-Rate Drop': 3,
        'QRS Widening (Conduction Delay)': 4,
        'Dynamic ST-Segment Changes': 5,
        'Severe Hyperkalemia Signs': 6,
        'Advanced Age': 7,
        'Male Sex': 8,
        'Underlying Cardiac Disease': 9,
        'Critical Illness (Sepsis/Shock)': 10    
    },
    'id2name': {
        1: 'Ventricular Tachyarrhythmias',
        2: 'Ventricular Ectopy / NSVT',
        3: 'Bradycardia or Heart-Rate Drop',
        4: 'QRS Widening (Conduction Delay))',
        5: 'Dynamic ST-Segment Changes',
        6: 'Severe Hyperkalemia Signs',
        7: 'Advanced Age',
        8: 'Male Sex',
        9: 'Underlying Cardiac Disease',
        10: 'Critical Illness (Sepsis/Shock)'
    }
}