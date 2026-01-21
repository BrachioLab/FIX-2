claim_grouping_template = """
You are an expert in XXX. You have a deep understanding of this subject. 
Your task is to behave like an XXX and identify which atomic claims are related to the given expert category.
We define "related" as claims that are topically relevant to the expert category and/or can be used to support the expert category.

Task description:
Input:
Output:

Here are some examples:

[Example 1]
[Example 2]
[Example 3]

Now identify which atomic claims are related to the given expert category:
CATEGORY: {}
CLAIMS: {}
"""

claim_grouping_politeness = """
You are an expert in politeness understanding. You have a deep understanding of this subject. 
Your task is to behave like an expert linguist and identify which atomic claims are related to the given expert category.
We define "related" as claims that are topically relevant to the expert category and/or can be used to support the expert category.

Task description:
Input: An expert psychologist's explanation of why a certain level of politeness might be attributed to an utterance, and a list of atomic claims.
Output: A list of atomic claims that are related to the given expert category.

"""

claim_grouping_massmaps = """
You are an expert in weak lensing mass maps understanding. You have a deep understanding of this subject. 
Your task is to behave like an expert cosmologist and identify which atomic claims are related to the given expert category.
We define "related" as claims that are topically relevant to the expert category and/or can be used to support the expert category.
For the weak lensing map we are analyzing, note that the number is below 0 if it shows up as between gray and blue, and 0 is gray, and between 0 and 2.9 is between gray and red, and above 2.9 is yellow. The numbers are in standard deviations of the mass map. Therefore, when you see a claim talking about "the presence of blue and gray", it means the presence of underdense areas in the map, and should therefore be included in the list of claims that are related to the given expert category that is related to underdense areas. On the other hand, if a claim talks about "the presence of red and yellow", it means the presence of overdense areas in the map, and should not be included in the list of claims that are related to the given expert category that is related to underdense areas.
Omega_m's value can be between 0.1 ~ 0.5, and sigma_8's value can be between 0.4 ~ 1.4.
We are using simulated weak lensing maps, which can have Omega_m and sigma_8 values of all kinds.


Task description:
Input: An expert cosmologist's explanation of why certain Omega_m and sigma_8 values were attributed to a weak lensing mass map, and a list of atomic claims.
Output: A list of atomic claims that are related to the given expert category. If there are no claims that are related to the given expert category, then the output should be "N/A".

Here are some examples:
Example 1:
CATEGORY: Lensing Peak (Cluster) Abundance: A higher count of prominent, high-convergence peaks in the map indicates a larger sigma_8, since a clumpier matter distribution produces more frequent massive halos.
CLAIMS:
The weak lensing map shows a mix of blue, gray, red, and some yellow regions.
The presence of blue and gray indicates underdense areas in the map.
The presence of red and yellow suggests overdense regions.
Yellow indicates significant mass concentrations or clusters.
The distribution and intensity of underdense and overdense regions being present and there are significant mass concentrations or clusters suggests a universe with moderate matter density and fluctuation levels.
The presence of several yellow regions, the significant mass concentrations or clusters, indicates a relatively high sigma_8.
The mix of blue and gray, the underdense areas, suggests a moderate Omega_m.

OUTPUT:
Yellow indicates significant mass concentrations or clusters.
The presence of several yellow regions, the significant mass concentrations or clusters, indicates a relatively high sigma_8.

Example 2:
CATEGORY: Void Size and Frequency: Extensive low-convergence void regions suggest a lower Omega_m, as a reduced overall matter density allows bigger underdense expanses to form in the cosmic web.
CLAIMS:
The weak lensing map shows a mix of blue, gray, red, and some yellow regions.
The presence of blue and gray indicates underdense areas in the map.
The presence of red and yellow suggests overdense regions.
Yellow indicates significant mass concentrations or clusters.
The distribution and intensity of underdense and overdense regions being present and there are significant mass concentrations or clusters suggests a universe with moderate matter density and fluctuation levels.
The presence of several yellow regions, the significant mass concentrations or clusters, indicates a relatively high sigma_8.
The mix of blue and gray, the underdense areas, suggests a moderate Omega_m.

OUTPUT:
The presence of blue and gray indicates underdense areas in the map.
The mix of blue and gray, the underdense areas, suggests a moderate Omega_m.

Example 3:
CATEGORY: Density Contrast Extremes: Very pronounced contrast between dense regions and empty voids - i.e. bright lensing peaks adjacent to dark void areas - signals an enhanced variance of the density field (high sigma_8), whereas subdued contrast suggests lower sigma_8.
CLAIMS:
The weak lensing map shows a mix of blue, gray, red, and some yellow regions.
The presence of blue and gray indicates underdense areas in the map.
The presence of red and yellow suggests overdense regions.
Yellow indicates significant mass concentrations or clusters.
The distribution and intensity of underdense and overdense regions being present and there are significant mass concentrations or clusters suggests a universe with moderate matter density and fluctuation levels.
The presence of several yellow regions, the significant mass concentrations or clusters, indicates a relatively high sigma_8.
The mix of blue and gray, the underdense areas, suggests a moderate Omega_m.

OUTPUT:
N/A

Now identify which atomic claims are related to the given expert category:
CATEGORY: {}
CLAIMS: {}
"""

claim_grouping_sepsis = """
You are a medical expert specializing in sepsis risk prediction. You have a deep understanding of this subject. 
Your task is to behave like an expert clinician and identify which atomic claims are related to the given expert category.
We define "related" as claims that are topically relevant to the expert category and/or can be used to support the expert category.

Task description:
Input: An expert sepsis clinician's explanation of why the patient is at high risk of developing sepsis within the next 12 hours, based on electronic health record (EHR) data collected during the first 2 hours of their emergency department (ED) admission, and a list of atomic claims.
Output: A list of atomic claims that are related to the given expert category. If there are no claims that are related to the given expert category, then the output should be "N/A".

Here are some examples:
Example 1:
CATEGORY: Elderly Susceptibility (Age ≥65 years): Advanced age (≥ 65 years) markedly increases susceptibility to rapid sepsis progression and higher mortality after infection.
CLAIMS:
The patient exhibits several risk factors for sepsis.
The patient is 71 years old.
A high triage temperature indicates fever.
OUTPUT:
The patient is 71 years old.

Example 2:
CATEGORY: SIRS Positivity (≥2 Criteria): Presence of ≥ 2 SIRS criteria—temperature > 38 °C or < 36 °C, heart rate > 90 bpm, respiratory rate > 20 /min or PaCO₂ < 32 mm Hg, or WBC > 12 000/µL or < 4 000/µL—identifies systemic inflammation consistent with early sepsis.
CLAIMS:
Another risk factor for sepsis is a high triage temperature.
A high triage temperature indicates fever.
Another risk factor for sepsis is an elevated respiratory rate.
OUTPUT:
Another risk factor for sepsis is a high triage temperature.
A high triage temperature indicates fever.
Another risk factor for sepsis is an elevated respiratory rate.

Example 3:
CATEGORY: Early Antibiotic/Culture Orders (within 2 hours): Administration of broad‑spectrum antibiotics or drawing of blood cultures within the first 2 hours signifies clinician suspicion of serious infection and should anchor sepsis risk assessment.
CLAIMS:
The patient exhibits several risk factors for sepsis.
The respiratory rate is 22.
The pulse oximetry is 92%.
OUTPUT:
N/A

Now identify which atomic claims are related to the given expert category:
CATEGORY: {}
CLAIMS: {}
"""
