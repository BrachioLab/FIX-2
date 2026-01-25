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
Output: 
RELATED CLAIMS: A list of atomic claims that are related to the given expert category that are copied verbatim from the input claims following the format in the examples. If there are no claims that are related to the given expert category, then the output should be "N/A". If a category is only related to void, then it should only contain claims that are related to void/underdense areas, and exclude claims that are about the relation between peaks/overdense areas and voids/underdense areas. The same goes for categories that are only related to peaks/overdense areas.
REASONING: A brief explanation of why you selected the claims that are related to the given expert category and why you judged the claims as you did.

Here are some examples:
Example 1:
CATEGORY: Lensing Peak (Cluster) Abundance - A higher count of prominent, high-convergence peaks in the map indicates a larger sigma_8, since a clumpier matter distribution produces more frequent massive halos.
CLAIMS:
The weak lensing map shows a mix of blue, gray, red, and some yellow regions.
The presence of blue and gray indicates underdense areas in the map.
The presence of red and yellow suggests overdense regions.
Yellow indicates significant mass concentrations or clusters.
The distribution and intensity of underdense and overdense regions being present and there are significant mass concentrations or clusters suggests a universe with moderate matter density and fluctuation levels.
The contrast between peaks and voids, along with the spatial distribution pattern, points to relatively moderate matter fluctuations.
The presence of several yellow regions, the significant mass concentrations or clusters, indicates a relatively high sigma_8.
The mix of blue and gray, the underdense areas, suggests a moderate Omega_m.

OUTPUT:
RELATED CLAIMS:
Yellow indicates significant mass concentrations or clusters.
The presence of several yellow regions, the significant mass concentrations or clusters, indicates a relatively high sigma_8.
REASONING:
These claims focus on the presence and multiplicity of significant mass concentrations, corresponding to prominent high-convergence peaks and their abundance, which directly reflects the notion of lensing peak or cluster count in this category. Claims describing underdense regions (blue and gray) are excluded because they pertain to voids rather than peaks. Claims about contrast between peaks and voids or overall distribution and intensity are excluded because they describe variance or relative structure rather than the number of clusters. More general statements about the presence of red and yellow regions are also excluded because they do not specifically emphasize peak abundance or cluster frequency.

Example 2:
CATEGORY: Void Size and Frequency - Extensive low-convergence void regions suggest a lower Omega_m, as a reduced overall matter density allows bigger underdense expanses to form in the cosmic web.
CLAIMS:
The weak lensing map shows a mix of blue, gray, red, and some yellow regions.
The presence of blue and gray indicates underdense areas in the map.
The presence of red and yellow suggests overdense regions.
Yellow indicates significant mass concentrations or clusters.
The distribution and intensity of underdense and overdense regions being present and there are significant mass concentrations or clusters suggests a universe with moderate matter density and fluctuation levels.
The contrast between peaks and voids, along with the spatial distribution pattern, points to relatively moderate matter fluctuations.
The presence of several yellow regions, the significant mass concentrations or clusters, indicates a relatively high sigma_8.
The mix of blue and gray, the underdense areas, suggests a moderate Omega_m.

OUTPUT:
RELATED CLAIMS:
The presence of blue and gray indicates underdense areas in the map.
The mix of blue and gray, the underdense areas, suggests a moderate Omega_m.
REASONING:
These claims describe the presence of underdense regions and relate the prevalence of blue and gray areas to the matter density parameter Omega_m, which reflects the existence and interpretation of void regions underlying this category. Claims describing overdense regions or clusters (red and yellow) are excluded because they pertain to peaks rather than voids. Claims about contrast between peaks and voids or overall distribution and intensity are excluded because they describe relative variance or structural contrast rather than the size or frequency of underdense regions themselves. Claims inferring sigma_8 are excluded because this category is concerned with matter density and void expansiveness rather than clustering amplitude.

Example 3:
CATEGORY: Density Contrast Extremes - Very pronounced contrast between dense regions and empty voids - i.e. bright lensing peaks adjacent to dark void areas - signals an enhanced variance of the density field (high sigma_8), whereas subdued contrast suggests lower sigma_8.
CLAIMS:
The weak lensing map shows a mix of blue, gray, red, and some yellow regions.
The presence of blue and gray indicates underdense areas in the map.
The presence of red and yellow suggests overdense regions.
Yellow indicates significant mass concentrations or clusters.
The distribution and intensity of underdense and overdense regions being present and there are significant mass concentrations or clusters suggests a universe with moderate matter density and fluctuation levels.
The contrast between peaks and voids, along with the spatial distribution pattern, points to relatively moderate matter fluctuations.
The presence of several yellow regions, the significant mass concentrations or clusters, indicates a relatively high sigma_8.
The mix of blue and gray, the underdense areas, suggests a moderate Omega_m.

OUTPUT:
RELATED CLAIMS:
The presence of blue and gray indicates underdense areas in the map.
The presence of red and yellow suggests overdense regions.
Yellow indicates significant mass concentrations or clusters.
The contrast between peaks and voids, along with the spatial distribution pattern, points to relatively moderate matter fluctuations.
REASONING:
These claims describe the presence of both overdense and underdense regions in the map and explicitly refer to the contrast between peaks and voids and the distribution and intensity of dense and empty areas. Together, they characterize how mass concentrations and void regions coexist and vary in strength across the map, which corresponds to the notion of density contrast and variance of the density field captured by this category.

Example 4:
CATEGORY: Connectivity of the Cosmic Web - A highly interconnected filament network (with filaments linking most clusters into a continuous web) hints at a higher Omega_m, whereas a more fragmented scene of isolated clumps separated by wide gaps is expected for a lower Omega_m.
CLAIMS:
The weak lensing map shows a mix of blue, gray, red, and some yellow regions.
The presence of blue and gray indicates underdense areas in the map.
The presence of red and yellow suggests overdense regions.
Yellow indicates significant mass concentrations or clusters.
The distribution and intensity of underdense and overdense regions being present and there are significant mass concentrations or clusters suggests a universe with moderate matter density and fluctuation levels.
The contrast between peaks and voids, along with the spatial distribution pattern, points to relatively moderate matter fluctuations.
The presence of several yellow regions, the significant mass concentrations or clusters, indicates a relatively high sigma_8.
The mix of blue and gray, the underdense areas, suggests a moderate Omega_m.

OUTPUT:
RELATED CLAIMS:
N/A
REASONING:
The claims are not related to the given expert category connectivity of the cosmic web because there are no claims that talk about the connectivity of the cosmic web and filaments.

Now identify which atomic claims are related to the given expert category:
CATEGORY: {}
CLAIMS: {}
"""

claim_grouping_cholec = """
You are an expert in laparoscopic cholecystectomy. You have a deep understanding of this subject.
Your task is to behave like an expert surgeon and identify which atomic claims are related to the given expert category.

We define "related" as claims that are topically relevant to the expert category and/or can be used to support the expert category.

Task description:
Input: An expert surgeon's explanation of why certain criteria were used to determine what is safe and unsafe in performing a laparoscopic cholecystectomy, and a list of atomic claims.
Output:
RELATED CLAIMS: A list of atomic claims that are related to the given expert category, copied verbatim from the input claims (one per line) following the format in the examples. If there are no claims related to the category, output "N/A".
REASONING: A brief explanation of (1) why the selected claims are related to the category (i.e., why they belong in this group) and (2) why key non-selected claims were excluded (e.g., they pertain to a different CVS criterion / different anatomical structure / general context but not evidence for this category).

Important guidelines:
- Only copy claims verbatim; do not rewrite claims.
- Prefer atomic, direct evidence claims for the category (e.g., visibility/identification of duct/artery, degree of clearance, detachment plane).
- Exclude claims that are about different categories even if they appear in the same scene (e.g., “two structures visible” vs “triangle cleared” vs “gallbladder detachment” are distinct).
- If a claim is purely general context (e.g., “the liver is on the left”) and does not support the category, exclude it.
- Include “risk/unsafe” claims only if they directly follow from (or explicitly reference) the category-specific deficiency (e.g., obscured Calot’s landmarks for “triangle cleared” / “inflammation bailout”).

Here are some examples:

Example 1:
CATEGORY: Calot's triangle cleared: Hepatocystic triangle must be fully cleared of fat/fibrosis so that its boundaries are unmistakable.
CLAIMS:
The liver parenchyma is evident on the left side of the image.
The gallbladder remnant is visible on the right side of the image.
Inflamed Calot's triangle tissue appears to be centrally located.
The Calot's triangle is bordered by the cystic duct inferiorly, common hepatic duct medially, and liver edge superiorly.
The tissue types visible include inflamed fibrous tissue, likely from chronic or acute cholecystitis, and smooth, reddish liver parenchyma.
The gallbladder tissue appears partially resected or necrotic.
There is evidence of scarring and possible adhesions.
The tissue in the central area looks thickened and fibrotic, obscuring normal anatomic planes.
Key unsafe zones include the area directly adjacent to the common bile duct and hepatic artery, as aberrant anatomy or inflammation here increases the risk of vascular or biliary injury.
The inflammation and scarring obscure key landmarks in Calot's triangle.
The obscured landmarks increase the risk of injuring the common bile duct and hepatic artery during laparoscopic cholecystectomy.

OUTPUT:
RELATED CLAIMS:
Inflamed Calot's triangle tissue appears to be centrally located.
The Calot's triangle is bordered by the cystic duct inferiorly, common hepatic duct medially, and liver edge superiorly.
The tissue types visible include inflamed fibrous tissue, likely from chronic or acute cholecystitis, and smooth, reddish liver parenchyma.
There is evidence of scarring and possible adhesions.
The tissue in the central area looks thickened and fibrotic, obscuring normal anatomic planes.
The inflammation and scarring obscure key landmarks in Calot's triangle.
REASONING:
These claims describe the condition and visibility of the hepatocystic (Calot’s) triangle—its anatomical boundaries and the presence of inflammation/scarring/fibrosis that can prevent full clearance—directly supporting whether the triangle is “cleared” and its boundaries are unmistakable. Claims about the liver or gallbladder position are excluded because they provide general scene context but do not establish clearance. Claims about “unsafe zones” or injury risk are excluded unless they specifically hinge on inadequate triangle clearance; here the more direct evidence is the obscured landmarks and thickened fibrotic tissue.

Example 2:
CATEGORY: Inflammation bailout: If dense scarring or distorted anatomy obscures Calot's triangle, convert to open surgery or a fenestrated subtotal approach rather than blind cutting.
CLAIMS:
The liver parenchyma is evident on the left side of the image.
The gallbladder remnant is visible on the right side of the image.
Inflamed Calot's triangle tissue appears to be centrally located.
The Calot's triangle is bordered by the cystic duct inferiorly, common hepatic duct medially, and liver edge superiorly.
The tissue types visible include inflamed fibrous tissue, likely from chronic or acute cholecystitis, and smooth, reddish liver parenchyma.
The gallbladder tissue appears partially resected or necrotic.
There is evidence of scarring and possible adhesions.
The tissue in the central area looks thickened and fibrotic, obscuring normal anatomic planes.
Key unsafe zones include the area directly adjacent to the common bile duct and hepatic artery, as aberrant anatomy or inflammation here increases the risk of vascular or biliary injury.
The inflammation and scarring obscure key landmarks in Calot's triangle.
The obscured landmarks increase the risk of injuring the common bile duct and hepatic artery during laparoscopic cholecystectomy.

OUTPUT:
RELATED CLAIMS:
Inflamed Calot's triangle tissue appears to be centrally located.
The tissue types visible include inflamed fibrous tissue, likely from chronic or acute cholecystitis, and smooth, reddish liver parenchyma.
There is evidence of scarring and possible adhesions.
The tissue in the central area looks thickened and fibrotic, obscuring normal anatomic planes.
The inflammation and scarring obscure key landmarks in Calot's triangle.
The obscured landmarks increase the risk of injuring the common bile duct and hepatic artery during laparoscopic cholecystectomy.
REASONING:
These claims collectively indicate severe inflammation/scarring and distorted planes that obscure Calot’s triangle landmarks, which is the key trigger for a bailout decision (avoid blind dissection and consider conversion or subtotal approach). Claims about simple anatomic location (liver/gallbladder position) are excluded because they do not establish obscured anatomy. Claims defining the triangle boundaries are excluded here because the bailout decision depends primarily on whether those boundaries are not safely identifiable due to scarring, rather than on reciting the standard anatomy.

Example 3:
CATEGORY: Only two structures visible: Only the cystic duct and cystic artery should be seen entering the gallbladder before any clipping or cutting.
CLAIMS:
The liver parenchyma is evident on the left side of the image.
The gallbladder remnant is visible on the right side of the image.
Inflamed Calot's triangle tissue appears to be centrally located.
The Calot's triangle is bordered by the cystic duct inferiorly, common hepatic duct medially, and liver edge superiorly.
The tissue types visible include inflamed fibrous tissue, likely from chronic or acute cholecystitis, and smooth, reddish liver parenchyma.
The gallbladder tissue appears partially resected or necrotic.
There is evidence of scarring and possible adhesions.
The tissue in the central area looks thickened and fibrotic, obscuring normal anatomic planes.
Key unsafe zones include the area directly adjacent to the common bile duct and hepatic artery, as aberrant anatomy or inflammation here increases the risk of vascular or biliary injury.
The inflammation and scarring obscure key landmarks in Calot's triangle.
The obscured landmarks increase the risk of injuring the common bile duct and hepatic artery during laparoscopic cholecystectomy.

OUTPUT:
RELATED CLAIMS:
N/A
REASONING:
None of the claims state that exactly two tubular structures (cystic duct and cystic artery) are visible entering the gallbladder, nor do they describe identification of these two structures prior to clipping/cutting. The claims focus on inflammation, scarring, triangle boundaries, and general risk zones rather than explicit confirmation of “two structures only,” so they do not provide evidence for this category.

Now identify which atomic claims are related to the given expert category:
CATEGORY: {}
CLAIMS: {}
"""

claim_grouping_sepsis = """
You are a medical expert specializing in sepsis risk prediction. You have a deep understanding of this subject. 
Your task is to behave like an expert clinician and identify which atomic claims are related to the given expert category.
We define "related" as claims that are topically relevant to the expert category and/or can be used to support the expert category.
If the claim explicitly states a condition, measurement, intervention, or fact that is part of the expert category definition, it must be included, even if it does not provide additional explanation or reasoning. 

Task description:
Input: An expert sepsis clinician's explanation of why the patient is at high risk of developing sepsis within the next 12 hours, based on electronic health record (EHR) data collected during the first 2 hours of their emergency department (ED) admission, and a list of atomic claims.
Output: 
RELATED CLAIMS: A list of atomic claims that are related to the given expert category that are copied verbatim from the input claims following the format in the examples. If there are no claims that are related to the given expert category, then the output should be "N/A". 
REASONING: A brief explanation of why the selected claims support the category and why key non-selected claims were excluded (e.g., they relate to a different category or provide only general context rather than evidence).

Here are some examples:
Example 1:
CATEGORY: Elderly Susceptibility (Age ≥65 years): Advanced age (≥ 65 years) markedly increases susceptibility to rapid sepsis progression and higher mortality after infection.
CLAIMS:
The patient exhibits several risk factors for sepsis.
The patient is 71 years old.
A high triage temperature indicates fever.

OUTPUT:
RELATED CLAIMS:
The patient is 71 years old.
REASONING:
“The patient is 71 years old” is included because it directly satisfies the category definition of elderly susceptibility (age ≥ 65), providing explicit evidence that the patient belongs to a higher-risk age group for sepsis progression. “The patient exhibits several risk factors for sepsis” is excluded because it is nonspecific and does not reference age or elderly status. “A high triage temperature indicates fever” is also excluded because it relates to physiologic signs of infection rather than age-based susceptibility and therefore does not support this category.

Example 2:
CATEGORY: SIRS Positivity (≥2 Criteria): Presence of ≥ 2 SIRS criteria—temperature > 38 °C or < 36 °C, heart rate > 90 bpm, respiratory rate > 20 /min or PaCO₂ < 32 mm Hg, or WBC > 12 000/µL or < 4 000/µL—identifies systemic inflammation consistent with early sepsis.
CLAIMS:
Another risk factor for sepsis is a high triage temperature.
A high triage temperature indicates fever.
Another risk factor for sepsis is an elevated respiratory rate.

OUTPUT:
RELATED CLAIMS:
Another risk factor for sepsis is a high triage temperature.
A high triage temperature indicates fever.
Another risk factor for sepsis is an elevated respiratory rate.
REASONING:
All three claims are included because they directly reference physiologic abnormalities that correspond to SIRS criteria. A high triage temperature and the presence of fever align with the SIRS temperature criterion (> 38 °C), while an elevated respiratory rate aligns with the SIRS respiratory rate criterion (> 20 /min). Although the claims are phrased generally and do not provide exact numeric values, they describe clinical features that are part of the SIRS definition and therefore are topically relevant to identifying SIRS positivity.

Example 3:
CATEGORY: Early Antibiotic/Culture Orders (within 2 hours): Administration of broad‑spectrum antibiotics or drawing of blood cultures within the first 2 hours signifies clinician suspicion of serious infection and should anchor sepsis risk assessment.
CLAIMS:
The patient exhibits several risk factors for sepsis.
The respiratory rate is 26.
The pulse oximetry is 92%.

OUTPUT:
RELATED CLAIMS:
N/A
REASONING:
No claims are included because none reference the administration of antibiotics or the collection of blood cultures within the first two hours. “The patient exhibits several risk factors for sepsis” is a nonspecific statement that does not indicate clinician actions or treatment decisions. “The respiratory rate is 26” and “The pulse oximetry is 92%” describe physiologic measurements, which may reflect illness severity but do not provide evidence of early antibiotic administration or blood culture orders. Therefore, none of the claims support the Early Antibiotic/Culture Orders category.

Example 4:
CATEGORY: High qSOFA Score (≥2): A qSOFA score ≥ 2 (respiratory rate ≥ 22 /min, systolic BP ≤ 100 mmHg, or altered mentation) flags high risk of sepsis‑related organ dysfunction and mortality.
CLAIMS:
The patient's vital signs indicate tachycardia.
The respiratory rate is 26.
A respiratory rate of 26 is concerning for sepsis.

OUTPUT:
RELATED CLAIMS:
The respiratory rate is 26.
A respiratory rate of 26 is concerning for sepsis.
REASONING:
The claims related to a respiratory rate of 26 are included because qSOFA explicitly uses respiratory rate ≥ 22/min as a criterion, and a rate of 26 meets this threshold and supports elevated qSOFA risk. The tachycardia claim is not included because heart rate is not part of the qSOFA criteria, even though it may be relevant in other sepsis assessment frameworks.

Example 5:
CATEGORY: Sepsis-Associated Hypotension (SBP <90 mmHg or MAP <70 mmHg, or ≥40 mmHg drop): Sepsis‑associated hypotension, defined as SBP < 90 mmHg, MAP < 70 mmHg, or a ≥ 40 mmHg drop from baseline, indicates progression toward septic shock.
CLAIMS:
The patient’s systolic blood pressure is 85 mmHg.
Mean arterial pressure is 65 mmHg.
Vasopressor support was initiated due to low blood pressure.

OUTPUT:
RELATED CLAIMS:
The patient’s systolic blood pressure is 85 mmHg.
Mean arterial pressure is 65 mmHg.
Vasopressor support was initiated due to low blood pressure.
REASONING:
All three claims are related because they support sepsis-associated hypotension and potential progression toward septic shock. An SBP of 85 mmHg meets the criterion of SBP < 90 mmHg, and a MAP of 65 mmHg meets MAP < 70 mmHg, both directly satisfying the category definition. The initiation of vasopressor support due to low blood pressure indicates clinically significant hypotension requiring escalation of hemodynamic support, which is consistent with worsening sepsis-related circulatory failure.

Now identify which atomic claims are related to the given expert category:
CATEGORY: {}
CLAIMS: {}
"""

claim_grouping_supernova = """
You are an astrophysics expert specializing in astrophysical classification. The possible classification labels are: RR Lyrae (RRL), peculiar Type Ia supernova (SN Ia-91bg), Type Ia supernova (SN Ia), superluminous supernova (SLSN-I), Type II supernova (SN II), microlensing single-lens (μLens-Single), eclipsing binary (EB), M-dwarf, kilonova (KN), tidal disruption event (TDE), peculiar Type Ia supernova (SN Iax), Type Ibc supernova (SN Ibc), Mira variable, and active galactic nucleus (AGN). You have a deep understanding of this subject. Your task is to behave like an expert astrophysicist and identify which atomic claims are related to the given expert category.
We define "related" as claims that are topically relevant to the expert category and/or can be used to support the expert category.

Task description:
Input: An astrophysics expert's explanation for why a particular multivariate time-series was classified into one of the above categories, and a list of atomic claims.
Output:
RELATED CLAIMS: A list of atomic claims that are related to the given expert category, copied verbatim from the input claims (one per line) following the format in the examples. If there are no claims related to the category, output "N/A".
REASONING: A brief explanation of why the selected claims support the category and why key non-selected claims were excluded (e.g., they relate to a different category or provide only general context rather than evidence).

Important guidelines:
- Only copy claims verbatim; do not rewrite claims.
- Prefer atomic, direct evidence claims for the category (e.g., visibility/identification of duct/artery, degree of clearance, detachment plane).
- Exclude claims that are about different categories even if they appear in the same scene (e.g., “two structures visible” vs “triangle cleared” vs “gallbladder detachment” are distinct).
- If a claim is purely general context (e.g., “the liver is on the left”) and does not support the category, exclude it.
- Include “risk/unsafe” claims only if they directly follow from (or explicitly reference) the category-specific deficiency (e.g., obscured Calot’s landmarks for “triangle cleared” / “inflammation bailout”).

Here are some examples:

Example 1:
CATEGORY: Contiguous non-zero flux: Contiguous non‑zero flux segments confirm genuine astrophysical activity and define the time windows from which transient features should be extracted.
CLAIMS:
The time series shows a rapid rise to a peak with subsequent decline.
A rapid rise to a peak with subsequent decline is characteristic of a Type Ia supernova light curve.
The pattern of observations was made at multiple wavelengths.

OUTPUT:
RELATED CLAIMS:
The time series shows a rapid rise to a peak with subsequent decline.
A rapid rise to a peak with subsequent decline is characteristic of a Type Ia supernova light curve.
REASONING:
The first two claims are related because they describe a continuous rise and decline in flux, which indicates a contiguous non-zero signal and genuine transient activity. This temporal structure is consistent with a real astrophysical event. The third claim is not related because multi-wavelength observation is a general property of the dataset and does not provide evidence of contiguous or sustained non-zero flux.

Example 2:
CATEGORY: Rise–decline rates: Characteristic rise‑and‑decline rates—such as the fast‑rise/slow‑fade morphology of many supernovae—encode energy‑release physics and serve as strong class discriminators.
CLAIMS:
The time series shows a rapid increase in brightness followed by a gradual decline.
The time series includes an initial flat phase followed by a sharp increase and consistent decline.
This rise and decline pattern is characteristic of type Ia supernovae.

OUTPUT:
RELATED CLAIMS:
The time series shows a rapid increase in brightness followed by a gradual decline.
The time series includes an initial flat phase followed by a sharp increase and consistent decline.
This rise and decline pattern is characteristic of type Ia supernovae.
REASONING:
All three claims are related because they directly describe the shape and rates of the light curve—fast rise, slower decline, and the presence of a plateau/flat phase before a sharp increase—which are exactly the rise–decline features the category targets. The third claim is also related because it explicitly links that rise/decline morphology to a discriminative class label (Type Ia), which is one of the main uses of rise–decline rates.

Example 3:
CATEGORY: Monotonic flux trends: Locally smooth, monotonic flux trends across one or multiple bands (plateaus, linear decays) capture physical evolution stages and help distinguish SN II‑P, SN II‑L, and related classes.
CLAIMS:
An initial flat phase followed by a sharp increase and decline.
Observations across multiple wavelengths are present.

OUTPUT:
RELATED CLAIMS:
An initial flat phase followed by a sharp increase and decline.
REASONING:
The first claim is related because it describes a locally smooth and monotonic phase (the initial flat plateau) followed by a coherent flux evolution, which reflects physical stages captured by monotonic flux trends. The second claim is not related because multi-wavelength observations are a general data property and do not describe monotonic behavior or flux evolution over time.

Example 4:
CATEGORY: Secondary maxima: Filter‑specific secondary maxima or shoulders in red/near‑IR bands—prominent in SNe Ia—are morphological features absent in most core‑collapse SNe.
CLAIMS:
The time series shows a rapid increase in brightness followed by a gradual decline.
The dataset represents a time series of observations for a astrophysical event.
Specific wavelengths such as 7545.98 Å, 8590.90 Å, and 9710.28 Å are present in the data.

OUTPUT:
RELATED CLAIMS:
N/A
REASONING:
No claims are included because none describe the presence of a secondary maximum or shoulder in specific red or near-infrared bands. The claims either describe a generic rise–decline pattern, provide dataset-level metadata, or merely list available wavelengths without identifying filter-specific secondary features.

Example 5:
CATEGORY: Event duration: Total event duration, measured from first detection to return to baseline, distinguishes short‑lived kilonovae and superluminous SNe from longer plateau or AGN variability phases.
CLAIMS:
The flux variability pattern is persistent over a long duration.
Activity across a wide range of wavelengths is typical for AGN emissions.
Irregular and persistent flux variability is characteristic of active galactic nuclei (AGN).

OUTPUT:
RELATED CLAIMS:
The flux variability pattern is persistent over a long duration.
Irregular and persistent flux variability is characteristic of active galactic nuclei (AGN).
REASONING:
The first claim is related because it directly states the event lasts a long time, which is exactly what event duration measures and uses for discrimination. The third claim is also related because it describes persistent variability as characteristic of AGN, tying a long-duration pattern to a specific class compared to shorter-lived transients. The second claim is not included because “activity across a wide range of wavelengths” is about spectral coverage, not how long the event persists from detection to baseline.

Now identify which atomic claims are related to the given expert category:
CATEGORY: {}
CLAIMS: {}
"""

claim_grouping_cardiac = """
You are a medical expert specializing in cardiac arrest prediction. 
You have a deep understanding of this subject. 
Your task is to behave like an expert clinician specializing in cardiac arrest and identify which atomic claims are related to the given expert category.
We define "related" as claims that are topically relevant to the expert category and/or can be used to support the expert category.
If the claim explicitly states a condition, measurement, intervention, or fact that is part of the expert category definition, it must be included, even if it does not provide additional explanation or reasoning. 

Task description:
Input: An expert cardiac arrest clinician's explanation of why the patient is at high risk of experiencing imminent cardiac arrest, based on patient background information and ECG time-series data collected in the initial ICU monitoring, and a set of atomic claims.

Output: 
RELATED CLAIMS: A list of atomic claims that are related to the given expert category that are copied verbatim from the input claims following the format in the examples. If there are no claims that are related to the given expert category, then the output should be "N/A". 
REASONING: A brief explanation of why you selected the claims that are related to the given expert category and why you judged the claims as you did.

Here are some examples:

[Example 1]
CATEGORY: Advanced Age: Increasing age is a major risk factor for cardiac arrest (events are very rare in patients under 30), with older ICU patients being significantly more prone to sudden arrest.
CLAIMS:
The patient's ECG graph shows significant irregularities with frequent and pronounced spikes and dips, indicating potential arrhythmic events.
The ECG patterns deviate from the normal consistent rhythm expected in a healthy heart.
The pronounced spikes on the ECG graph, particularly prominent around the 60 to 120-second marks, could signify ventricular tachycardia or fibrillation.
The patient is young.
The primary risk factor stems from trauma-induced complications from a motor vehicle collision.
Trauma-induced complications such as cardiac tamponade or myocardial contusion contribute to the prediction of high cardiac risk.

OUTPUT:
RELATED CLAIMS:
The patient is young.
REASONING:
This claim is related because it directly comments on the patient's age as a factor for why they may or may not experience imminent cardiac arrest, whereas the other claims do not relate to age.

[Example 2]
CATEGORY: Severe Hyperkalemia Signs: Electrocardiographic signs of severe hyperkalemia (such as peaked T-waves, loss of P-waves, and a widening QRS complex) herald an impending arrest – as potassium levels rise, the ECG may evolve to a sine-wave pattern and typically culminate in ventricular fibrillation or asystole without immediate intervention. Hyperkalemia is a frequent cause of in-hospital cardiac arrest especially among patients on dialysis / end stage renal disease. Looking for signs of hyperkalemia can be important to understand risk of cardiac arrest, especially in selected populations.
CLAIMS:
The patient's ECG graph shows significant irregularities with frequent and pronounced spikes and dips, indicating potential arrhythmic events.
The ECG patterns deviate from the normal consistent rhythm expected in a healthy heart.
The pronounced spikes on the ECG graph, particularly prominent around the 60 to 120-second marks, could signify ventricular tachycardia or fibrillation.
The patient is young.
The primary risk factor stems from trauma-induced complications from a motor vehicle collision.
Trauma-induced complications such as cardiac tamponade or myocardial contusion contribute to the prediction of high cardiac risk.

OUTPUT:
RELATED CLAIMS:
N/A
REASONING:
None of the provided claims mention electrocardiographic features specific to severe hyperkalemia (such as peaked T-waves, loss of P-waves, widening QRS complexes, or sine-wave patterns), nor do they reference elevated potassium levels, renal failure, or dialysis. The claims describe general ECG irregularities, arrhythmias (e.g., ventricular tachycardia or fibrillation), age, and trauma-related causes, which are not specific indicators of hyperkalemia-related cardiac arrest risk. Therefore, no claim directly supports or relates to the expert category of severe hyperkalemia signs.

Now identify which atomic claims are related to the given expert category:
CATEGORY: {}
CLAIMS: {}
"""

# alignment_cardiac = """You will be given a single claim explaining why a patient was predicted to be at high or low risk of experiencing cardiac arrest within the next {} (Yes/No). You will also be given a series of categories that an expert clinician would use to perform determine if there is high risk of imminent cardiac arrest.

# Your task is as follows:
# 1. Determine which expert category is most aligned with the claim. 
# 2. Rate how strongly the category aligns with the claim. Choose from complete, partial, or none.

# Category explanations:
# Complete: The claim is specific, directly relevant, and fully captures the meaning and intent of the expert category.
# Partial: The claim partially refers to the expert category but lacks key details, uses vague language, is overly general, or contains noise.
# None: The claim references something unrelated to the expert category, or misinterprets the category's meaning.

# Return your answer as:
# Category: <category>
# Category ID: <category ID>
# Category Alignment Rating: <complete/partial/none>
# Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

# -----
# Expert categories:
# 1. Ventricular Tachyarrhythmias: Ventricular tachyarrhythmias refer to abnormally rapid heart beats originating from the lower chambers (ventricles) of the heart. A common type is ventricular tachycardia, which can either be tolerated by the patient (i.e. patient is not arresting) or lead to hemodynamic collapse (i.e. state of low blood pressure, unconsciousness, and cardiac arrest).
# 2. Ventricular Ectopy / NSVT: Runs of non-sustained ventricular tachycardia (NSVT) or frequent premature ventricular contractions may indicate electrical instability in critically ill patients, particularly those with underlying coronary disease or cardiomyopathy. While NSVT is generally considered a more benign form of ventricular tachyarrhythmia, its presence reflects transient abnormal rhythms originating from the lower chambers and does not necessarily always signal impending cardiac arrest.
# 3. Bradycardia or Heart-Rate Drop: The onset of significant bradycardia or a sudden ≥30% decline in heart rate is a well-documented precursor to in-hospital cardiac arrest (often preceding pulseless electrical activity or asystole) and should be treated as an alarm sign
# 4. QRS Widening (Conduction Delay): New or progressive prolongation of the QRS duration on the ECG is an ominous finding in the ICU, often observed in the minutes before cardiac arrest and associated with higher mortality due to deteriorating ventricular conduction
# 5. Dynamic ST-Segment Changes: Acute ischemic changes on continuous ECG (notably ST-segment elevation or depression) signal low blood flow in the coronary arteries, indicating myocardial infarction or injury, and may precede imminent ventricular fibrillation or cardiac arrest. As blood flow continues to decrease, the ischemic heart can fibrillate or go into VT/VF (i.e. cardiac arrest). Although ST segment changes are common, the link to cardiac arrest is rare but possible.
# 6. Severe Hyperkalemia Signs: Electrocardiographic signs of severe hyperkalemia (such as peaked T-waves, loss of P-waves, and a widening QRS complex) herald an impending arrest – as potassium levels rise, the ECG may evolve to a sine-wave pattern and typically culminate in ventricular fibrillation or asystole without immediate intervention. Hyperkalemia is a frequent cause of in-hospital cardiac arrest especially among patients on dialysis / end stage renal disease. Looking for signs of hyperkalemia can be important to understand risk of cardiac arrest, especially in selected populations.
# 7. Advanced Age: Increasing age is a major risk factor for cardiac arrest (events are very rare in patients under 30), with older ICU patients being significantly more prone to sudden arrest
# 8. Male Sex: Male gender is associated with a higher incidence of cardiac arrest, as most cardiac arrests occur in men (with women’s risk rising post-menopause).
# 9. Underlying Cardiac Disease: The presence of serious cardiac conditions – such as coronary artery disease (especially a recent myocardial infarction) or severe heart failure – greatly elevates short-term cardiac arrest risk by creating an electrically and hemodynamically unstable myocardium.
# 10. Critical Illness (Sepsis/Shock): Severe sepsis or septic shock substantially raises the likelihood of cardiac arrest in the near term by causing hypoxia, hypotension, and metabolic derangements that often lead to pulseless electrical activity or asystole.
# -----

# Here are some examples:
# [Example 1]
# Claim: A skin lesion of the scalp is a condition not directly related to cardiac function.
# Category: Critical Illness (Sepsis/Shock)
# Category ID: 10
# Category Alignment Rating: none
# Reasoning: While a scalp lesion is not directly cardiac-related, if interpreted as a possible sign of infection or systemic compromise (e.g., an infected wound in a septic patient), it could weakly align with the critical illness category. However, without explicit signs of sepsis or shock, the connection remains speculative, hence the low alignment rating.

# [Example 2]
# Claim: The irregularity in the ECG could indicate a dangerous arrhythmia, such as ventricular tachycardia or fibrillation.
# Category: Ventricular Tachyarrhythmias
# Category ID: 1
# Category Alignment Rating: complete
# Reasoning: The claim directly references dangerous arrhythmias such as ventricular tachycardia and fibrillation, which are hallmark indicators of the Extreme Tachyarrhythmias category. These arrhythmias are known precursors to sudden cardiac arrest. While the claim does not specify the duration or ventricular dysfunction context, the alignment is still very strong due to the mention of the precise arrhythmias characteristic of this category.

# [Example 3]
# Claim: The ECG irregularities are suggested by inconsistent waveform intervals and amplitudes.
# Category: Ventricular Ectopy / NSVT
# Category ID: 2
# Category Alignment Rating: partial
# Reasoning: The claim points to ECG irregularities—specifically inconsistent waveform intervals and amplitudes—which can reflect transient abnormal ventricular rhythms such as NSVT or frequent PVCs. These patterns often manifest with variable intervals and amplitudes but are nonspecific. While the claim lacks precision, it aligns moderately with ventricular ectopy as a potential explanation for instability.

# [Example 4]
# Claim: The patient is 86 years old. 
# Category: Advanced Age
# Category ID: 7
# Category Alignment Rating: complete
# Reasoning: The claim directly references advanced age, which this category identifies as a major risk factor for cardiac arrest.

# [Example 5]
# Claim: The admission wasn't due to cardiac issues.
# Category: Underlying Cardiac Disease
# Category ID: 9
# Category Alignment Rating: partial
# Reasoning: The claim highlights the absence of underlying cardiac disease, which is the inverse of a known risk factor for cardiac arrest. While the expert category focuses on increased risk due to the presence of cardiac disease, this claim indirectly relates to it by implying a potentially lower risk. The alignment is moderate because the claim addresses the category by exclusion rather than direct evidence of risk.

# Now, determine the category and alignment rating for the following claim:
# Claim: {}
# """