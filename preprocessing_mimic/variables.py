# Definition of all included icu stations.
m_stations = ["Cardiac Surgery", "Cardiac Vascular Intensive Care Unit (CVICU)", "Cardiology",
              "Cardiology Surgery Intermediate", "Coronary Care Unit (CCU)", "Emergency Department",
              "Emergency Department Observation", "Hematology/Oncology", "Hematology/Oncology Intermediate",
              "Labor & Delivery", "Medical Intensive Care Unit (MICU)", "Medical/Surgical (Gynecology)",
              "Medical/Surgical Intensive Care Unit (MICU/SICU)", "Medicine", "Medicine/Cardiology",
              "Medicine/Cardiology Intermediate", "Med/Surg", "Med/Surg/GYN", "Med/Surg/Trauma",
              "Neonatal Intensive Care Unit (NICU)", "Neuro Intermediate", "Neurology", "Neuro Stepdown",
              "Neuro Surgical Intensive Care Unit (Neuro SICU)", "Nursery - Well Babies", "Observation",
              "Obstetrics Antepartum", "Obstetrics Postpartum", "Obstetrics (Postpartum & Antepartum)", "PACU",
              "Psychiatry", "Special Care Nursery (SCN)", "Surgery", "Surgery/Pancreatic/Biliary/Bariatric",
              "Surgery/Trauma", "Surgery/Vascular/Intermediate", "Surgical Intensive Care Unit (SICU)",
              "Thoracic Surgery", "Transplant", "Trauma SICU (TSICU)", "Unknown", "Vascular"]
# Information derived from
# https://www.bidmc.org/patient-and-visitor-information/adult-icu/meet-our-icu-care-team-and-leadership
# and info of clinic for "Anesthesia, Critical Care and Pain Medicine"
# https://www.bidmc.org/centers-and-departments/anesthesia-critical-care-and-pain-medicine/programs-and-services/
# critical-care
m_icus = [
    # Anesthesia, Critical Care and Pain Medicine
    "Surgical Intensive Care Unit (SICU)",
    "Trauma SICU (TSICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
    # Mixed Anesthesia, Critical Care and Pain Medicine and Division of Pulmonary, Critical Care and Sleep Medicine
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    # Division of Pulmonary, Critical Care and Sleep Medicine
    "Medical Intensive Care Unit (MICU)",
    # Cardiology
    "Coronary Care Unit (CCU)",
    # Pediatrics (excluded anyway because only adult patients)
    "Neonatal Intensive Care Unit (NICU)"
]
# IMC stations derived from names.
m_im_icus = [
    "Neuro Intermediate",
    "Neuro Stepdown",
    "Cardiology Surgery Intermediate",
    "Hematology/Oncology Intermediate",
    "Medicine/Cardiology Intermediate",
    "Surgery/Vascular/Intermediate"
]
# Only include stations only managed by Anesthesia, Critical Care and Pain Medicine.
m_included_stations = [
    "Surgical Intensive Care Unit (SICU)",
    "Trauma SICU (TSICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)"
]
m_anesthesiology_stations = m_included_stations
# Define normal ward station.
# Exclude post anesthesia care unit because it indicates surgery, i.e. planned readmission.
# Cardiac surgery seems to be a ward and no operating room, so included.
m_normal_stations = [s for s in m_stations if
                     ((s not in m_icus) and (s not in m_im_icus) and (s != 'PACU') and (s != 'Unknown'))]
