category_alignment_template = """You will be given [description of claims and expert category]

Your task is as follows:
Rate how strongly the set of claims align with the category. Choose from complete, partial, or none.

Alignment explanations:
Complete: The claim is specific, directly relevant, and fully captures the meaning and intent of the expert category.
Partial: The claim partially refers to the expert category but lacks key details, uses vague language, is overly general, or contains noise.
None: The claim references something unrelated to the expert category, or misinterprets the category's meaning

Return your answer as:
Reasoning: <A brief explanation of why you judged the alignment rating as you did.>
Category Alignment Rating: <rating>

Here are some examples:
[Example 1]
[Example 2]
[Example 3]

Now, determine the alignment rating for the following expert category and set of claims:
Category: {}
Claims: {}
"""

category_alignment_politeness = """You will be given a set of claims that relate to why a politeness rating was given to an utterance, and a series of categories that an expert linguist would use to perform this type of politeness classification.

Your task is as follows:
Rate how strongly the set of claims align with the category. Choose from complete, partial, or none.

Alignment explanations:
Complete: The claim is specific, directly relevant, and fully captures the meaning and intent of the expert category.
Partial: The claim partially refers to the expert category but lacks key details, uses vague language, is overly general, or contains noise.
None: The claim references something unrelated to the expert category, or misinterprets the category's meaning.
"""

category_alignment_massmaps = """You will be given a set of claims that relate to why predictions for Omega_m and sigma_8 values were given to a weak lensing mass map. You will also be given a series of categories that an expert cosmologist would use to perform this type of cosmological parameter prediction.

Your task is as follows:
Rate how strongly the set of claims align with the category. Choose from complete, partial, or none.

Alignment explanations:
Complete: The claim is specific, directly relevant, and fully captures the meaning and intent of the expert category.
Partial: The claim partially refers to the expert category but lacks key details, uses vague language, is overly general, or contains noise.
None: The claim references something unrelated to the expert category, or misinterprets the category's meaning. If the claims explicitly conclude the opposite parameter direction than the category (e.g., say lower Omega_m where the category implies higher Omega_m), rate none even if the visual description matches.

Return your answer as:
Reasoning: <A brief explanation of why you judged the alignment rating as you did.>
Category Alignment Rating: <rating>

Here are some examples:
Example 1:
INPUT:
Category: Lensing Peak (Cluster) Abundance - A higher count of prominent, high-convergence peaks in the map indicates a larger sigma_8, since a clumpier matter distribution produces more frequent massive halos.
Claims:
Yellow indicates significant mass concentrations or clusters.
The presence of several yellow regions, the significant mass concentrations or clusters, indicates a relatively high sigma_8.

OUTPUT:
Reasoning: The claim directly talks about a large number of yellow regions (high-convergence peaks) in the map, and how it indicates high sigma_8. This aligns with the Lensing Peak (Cluster) Abundance category which says a large number of peaks / clusters indicates a larger sigma_8.
Category Alignment Rating: complete

Example 2:
INPUT:
Category: Void Size and Frequency - Extensive low-convergence void regions suggest a lower Omega_m, as a reduced overall matter density allows bigger underdense expanses to form in the cosmic web.
Claims:
The presence of blue and gray indicates underdense areas in the map.
The mix of blue and gray, the underdense areas, suggests a moderate Omega_m.

OUTPUT:
Reasoning: The claims identify the presence of underdense (blue and gray) regions, which is topically related to voids in the weak lensing map. However, the expert category specifically concerns the size and extensiveness of void regions and their implication for a low Omega_m. The claims do not describe the voids as large or extensive, and they interpret the underdense regions as indicating a moderate rather than low Omega_m. Therefore, while the claims are related to voids in a general sense, they do not capture the expert mechanism of extensive voids implying low Omega_m.
Category Alignment Rating: partial

Example 3:
INPUT:
Category: Density Contrast Extremes - Very pronounced contrast between dense regions and empty voids - i.e. bright lensing peaks adjacent to dark void areas - signals an enhanced variance of the density field (high sigma_8), whereas subdued contrast suggests lower sigma_8.
Claims:
N/A

OUTPUT:
Reasoning: There is no claim and thus no alignment with the expert category.
Category Alignment Rating: none

Example 4:
INPUT:
Category: Connectivity of the Cosmic Web - A highly interconnected filament network … hints at a higher Omega_m …
Claims:
There is a highly interconnected filament network in the map.
A highly interconnected filament network strongly indicates a lower Omega_m.

OUTPUT:
Reasoning: The claims correctly identify an interconnected filament network, but they explicitly infer the opposite Omega_m direction from the expert category. Because the category states interconnectedness supports higher Omega_m (not lower), the claims misinterpret the category rather than partially matching it.
Category Alignment Rating: none

Example 5:
INPUT:
Category: Filament Thickness and Sharpness - Bold, sharply defined filaments threading between clusters imply a higher sigma_8 (stronger small-scale clustering), whereas thin or diffuse filaments point to a lower amplitude of matter fluctuations.
Claims:
The map shows several filaments connecting dense regions, some of which appear well defined.
Some filaments appear relatively thick and well defined, while others are faint and diffuse.
The mixture of thick and thin filaments suggests moderate clustering strength.
The map also contains multiple underdense void regions.

OUTPUT:
Reasoning: The claims refer to filament thickness and sharpness, which are directly relevant to the expert category. However, the description mixes thick and diffuse filaments and concludes only moderate clustering strength, rather than clearly linking bold, sharply defined filaments to high sigma_8. The inclusion of unrelated void information further adds noise. As a result, the claims partially reflect the expert mechanism but do not cleanly or strongly support it.
Category Alignment Rating: partial

Now, determine the alignment rating for the following expert category and set of claims:
Category: {}
Claims: {}
"""

category_alignment_cholec = """You will be given a set of claims that relate to why a proposed "safe" or "unsafe" zone was identified in laparoscopic cholecystectomy. You will also be given a series of categories that an expert surgeon would use to assess surgical safety.

Your task is as follows:
Rate how strongly the set of claims align with the category. Choose from complete, partial, or none.

Alignment explanations:
Complete: The claims are specific, directly relevant, and fully capture the meaning and intent of the expert category.
Partial: The claims partially refer to the expert category but lack key details, use vague language, are overly general, or contain noise.
None: The claims reference something unrelated to the expert category, misinterpret the category’s meaning, or explicitly conclude the opposite of what the category implies.

Return your answer as:
Reasoning: <A brief explanation of why you judged the alignment rating as you did.>
Category Alignment Rating: <complete/partial/none>

Here are some examples:

Example 1:
INPUT:
Category: Calot's triangle cleared - Hepatocystic triangle must be fully cleared of fat/fibrosis so that its boundaries are unmistakable.
Claims:
Inflamed Calot's triangle tissue appears to be centrally located.
There is evidence of scarring and possible adhesions.
The tissue in the central area looks thickened and fibrotic, obscuring normal anatomic planes.
The inflammation and scarring obscure key landmarks in Calot's triangle.

OUTPUT:
Reasoning: The claims directly describe inflammation, scarring, and obscured landmarks within Calot's triangle, indicating that the triangle is not clearly cleared and that its boundaries are not unmistakable. These observations precisely reflect the expert concern addressed by this category.
Category Alignment Rating: complete

--------------------------------------------------

Example 2:
INPUT:
Category: Inflammation bailout - If dense scarring or distorted anatomy obscures Calot's triangle, convert to open surgery or a fenestrated subtotal approach rather than blind cutting.
Claims:
The tissue types visible include inflamed fibrous tissue, likely from chronic or acute cholecystitis.
There is evidence of scarring and possible adhesions.
The tissue in the central area looks thickened and fibrotic, obscuring normal anatomic planes.
The obscured landmarks increase the risk of injuring the common bile duct and hepatic artery during laparoscopic cholecystectomy.

OUTPUT:
Reasoning: The claims describe dense inflammation, scarring, distorted planes, and increased injury risk, which directly correspond to the expert rationale for abandoning standard dissection and performing a bailout procedure. The claims fully capture the intent of this category.
Category Alignment Rating: complete

--------------------------------------------------

Example 3:
INPUT:
Category: Only two structures visible - Only the cystic duct and cystic artery should be seen entering the gallbladder before any clipping or cutting.
Claims:
The Calot's triangle is bordered by the cystic duct inferiorly, common hepatic duct medially, and liver edge superiorly.
Inflamed Calot's triangle tissue appears to be centrally located.
The gallbladder remnant is visible on the right side of the image.

OUTPUT:
Reasoning: The claims describe general anatomy, tissue condition, and gallbladder presence but do not mention any structures entering the gallbladder, their number, or their identity. Since the claims provide no evidence relevant to the two-structure visibility requirement, there is no alignment with this category.
Category Alignment Rating: none

--------------------------------------------------

Example 4:
INPUT:
Category: Above the R4U line - Dissection must remain cephalad to an imaginary line from Rouviere's sulcus to liver segment IV umbilical fissure to avoid the common bile duct.
Claims:
Key unsafe zones include the area directly adjacent to the common bile duct and hepatic artery, as aberrant anatomy or inflammation here increases the risk of vascular or biliary injury.
The obscured landmarks increase the risk of injuring the common bile duct and hepatic artery during laparoscopic cholecystectomy.

OUTPUT:
Reasoning: The claims correctly identify injury risk near the common bile duct, but they do not mention Rouviere's sulcus, the R4U line, or the requirement to remain cephalad to this landmark. The claims are related to biliary injury risk but do not capture the specific spatial safety rule of this category.
Category Alignment Rating: partial

--------------------------------------------------

Example 5:
INPUT:
Category: Cystic lymph node (calot's node) guide - Identify the cystic lymph node and clip the artery on the gallbladder side of the node to avoid injuring the hepatic artery.
Claims:
N/A

OUTPUT:
Reasoning: There are no claims provided and therefore no information that could align with the expert category.
Category Alignment Rating: none

Now, determine the alignment rating for the following expert category and set of claims:
Category: {}
Claims: {}
"""

category_alignment_sepsis = """You will be given a set of claims explaining why a patient was predicted to be at high or low risk of sepsis within the next 12 hours (Yes/No). You will also be given a series of categories that an expert clinician would use to perform this type of sepsis prediction.

Your task is as follows:
Rate how strongly the set of claims align with the category. Choose from complete, partial, or none.

Alignment explanations:
Complete: The claim is specific, directly relevant, and fully captures the meaning and intent of the expert category.
Partial: The claim partially refers to the expert category but lacks key details, uses vague language, is overly general, or contains noise.
None: The claim references something unrelated to the expert category, or misinterprets the category's meaning

Return your answer as:
Reasoning: <A brief explanation of why you judged the alignment rating as you did.>
Category Alignment Rating: <rating>

Here are some examples:
Example 1:
INPUT:
Category: Elderly Susceptibility (Age ≥65 years): Advanced age (≥ 65 years) markedly increases susceptibility to rapid sepsis progression and higher mortality after infection.
Claims:
The patient is 71 years old.

OUTPUT:
Reasoning: The claim directly states that the patient is 71 years old, which satisfies the category’s defining criterion (age ≥ 65) and is therefore specifically and fully aligned with “Elderly Susceptibility.”
Category Alignment Rating: complete
"""