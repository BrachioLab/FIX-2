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
Every atomic claim must be assigned to at least one expert category. Claims may appear under multiple categories if applicable, but no atomic claim should be left unassigned to all categories.

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
RELATED CLAIMS:
The patient is 71 years old.
REASONING:
This category is specifically triggered by age ≥ 65. The claim “The patient is 71 years old” directly meets the criterion and supports elderly susceptibility to rapid sepsis progression and higher mortality risk. The other claims (general sepsis risk factors and fever) may relate to sepsis risk overall but do not establish the age-based threshold, so they are not necessary for this category.

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
The claims indicate the presence of fever through a high triage temperature, which satisfies the SIRS temperature criterion (> 38 °C), and they also state that the patient has an elevated respiratory rate, satisfying the SIRS respiratory rate criterion (> 20 /min). Together, these constitute at least two SIRS criteria. Therefore, the related claims collectively support SIRS positivity, indicating systemic inflammation consistent with early sepsis.

Example 3:
CATEGORY: Early Antibiotic/Culture Orders (within 2 hours): Administration of broad‑spectrum antibiotics or drawing of blood cultures within the first 2 hours signifies clinician suspicion of serious infection and should anchor sepsis risk assessment.
CLAIMS:
The patient exhibits several risk factors for sepsis.
The respiratory rate is 22.
The pulse oximetry is 92%.

OUTPUT:
RELATED CLAIMS:
N/A
REASONING:
The category requires explicit evidence that broad-spectrum antibiotics were administered or blood cultures were drawn within the first 2 hours, indicating clinician suspicion of serious infection. The provided claims describe abnormal clinical findings (elevated respiratory rate and reduced oxygen saturation) and a general statement about sepsis risk factors, but they do not mention any antibiotic administration or blood culture orders.


Now identify which atomic claims are related to the given expert category:
CATEGORY: {}
CLAIMS: {}
"""