decomposition_template = """
You will be given a paragraph that explains XXX. Your task is to decompose this explanation into individual claims that are:

Atomic: Each claim should express only one clear idea or judgment.
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph.
Faithful: The claims must preserve the original meaning, nuance, and tone.

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:

INPUT:

OUTPUT:

Now decompose the following paragraph into atomic, standalone claims:
INPUT: {}
"""

decomposition_politeness = """
You will be given a paragraph that explains why a certain level of politeness was attributed to an utterance. Your task is to decompose this explanation into individual claims that are:

Atomic: Each claim should express only one clear idea or judgment.
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph.
Faithful: The claims must preserve the original meaning, nuance, and tone. Do not omit hedging language (e.g., "seems to," "somewhat," "lacks overt markers") or subjective phrasing if present.

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:

INPUT: This utterance is formal and professional, with no overtly rude language. The phrasing is neutral-to-polite, as it avoids accusatory or dismissive tones. The use of "I am copying them here" is transparent and non-confrontational, and "seem to constitute" softens any potential imposition by acknowledging some level of subjectivity. However, it lacks explicit politeness markers such as "please" or "thank you," which would elevate it to "extremely polite."

OUTPUT:
The utterance is formal and professional.
The utterance contains no overtly rude language.
The phrasing is neutral-to-polite because it avoids accusatory or dismissive tones.
The phrase "I am copying them here" is transparent and non-confrontational.
The phrase "seem to constitute" softens any potential imposition by acknowledging subjectivity.
The utterance lacks explicit politeness markers such as "please" or "thank you."
The lack of explicit politeness markers prevents the utterance from being considered "extremely polite."

Now decompose the following paragraph into atomic, standalone claims:
INPUT: {}
"""

decomposition_massmaps = """
You will be given a paragraph that explains why certain Omega_m and sigma_8 values were attributed to a weak lensing mass map. Your task is to decompose this explanation into individual claims that are:

Atomic: Each claim should express only one clear idea or judgment.
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph. When a claim talks about how there is a significant part of blue region, it should be rewritten to include the information that blue regions are underdense areas so that it can be understood outside the context. When a claim talks about "the level of contrast indicates ..." with a sentence preceding it saying "the contrast is high", it should be rewritten to include the information: "the level of contrast being high indicates ...". On the other hand, if "the contrast is high" is not indicated in the previous sentence, then it should not be included in the claim as we want to include only the information that is directly indicated in the paragraph.
Faithful: The claims must preserve the original meaning, nuance, and tone. Do not omit hedging language (e.g., "seems to," "somewhat," "lacks overt markers") or subjective phrasing if present.

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:

INPUT: The weak lensing map shows a mix of blue, gray, red, and some yellow regions. The presence of blue and gray indicates underdense areas, while red and yellow suggest overdense regions, with yellow indicating significant mass concentrations or clusters. The distribution and intensity of these colors suggest a universe with moderate matter density and fluctuation levels. The presence of several yellow regions indicates a relatively high sigma_8, while the mix of blue and gray suggests a moderate Omega_m.

OUTPUT:
The weak lensing map shows a mix of blue, gray, red, and some yellow regions.
The presence of blue and gray indicates underdense areas in the map.
The presence of red and yellow suggests overdense regions.
Yellow indicates significant mass concentrations or clusters.
The distribution and intensity of underdense and overdense regions being present and there are significant mass concentrations or clusters suggests a universe with moderate matter density and fluctuation levels.
The presence of several yellow regions, the significant mass concentrations or clusters, indicates a relatively high sigma_8.
The mix of blue and gray, the underdense areas, suggests a moderate Omega_m.

Now decompose the following paragraph into atomic, standalone claims:
INPUT: {}
"""

decomposition_cholec = """
You will be given a paragraph that provides a detailed explanation of a specific step, observation, or concept in laparoscopic cholecystectomy.

Your task is to decompose this explanation into individual claims that are:
- Specific to cholecystectomy: Each claim should pertain directly to the laparoscopic gallbladder removal procedure.
- Atomic: Each claim expresses a single, clear idea or judgment.
- Standalone: Each claim is self-contained and understandable without requiring the original paragraph.
- Faithful: Each claim preserves the original meaning, nuance, and tone of the paragraph.

Here is an example of how to format your output:

------

INPUT:
In this laparoscopic view, the primary anatomical landmarks evident include the liver parenchyma on the left side, the gallbladder remnant on the right, and what appears to be inflamed Calot's triangle tissue centrally located. The Calot's triangle is bordered by the cystic duct (inferiorly), common hepatic duct (medially), and the liver edge (superiorly). Safe operative dissection typically occurs within this triangle after clear identification and isolation of the cystic duct and cystic artery.

The tissue types visible include inflamed fibrous tissue, likely from chronic or acute cholecystitis, and smooth, reddish liver parenchyma. The gallbladder tissue appears partially resected or necrotic, and there is evidence of scarring and possible adhesions. The tissue in the central area looks thickened and fibrotic, obscuring normal anatomic planes, making this an unsafe region for blind dissection without clear identification of structures.

Key unsafe zones include the area directly adjacent to the common bile duct and hepatic artery, as aberrant anatomy or inflammation here increases the risk of vascular or biliary injury. Safe dissection requires careful blunt dissection and possibly intraoperative cholangiography to delineate anatomy before proceeding.

I provided this analysis based on the visual signs of inflammation and scarring, which obscure key landmarks in Calot's triangle, increasing the risk of injuring the common bile duct or hepatic artery during laparoscopic cholecystectomy.


OUTPUT:
The liver parenchyma is evident on the left side of the image.
The gallbladder remnant is visible on the right side of the image.
Inflamed Calot's triangle tissue appears to be centrally located.
The Calot's triangle is bordered by the cystic duct inferiorly, common hepatic duct medially, and liver edge superiorly.
Safe operative dissection typically occurs within this triangle after clear identification and isolation of the cystic duct and cystic artery.
The tissue types visible include inflamed fibrous tissue, likely from chronic or acute cholecystitis, and smooth, reddish liver parenchyma.
The gallbladder tissue appears partially resected or necrotic.
There is evidence of scarring and possible adhesions.
The tissue in the central area looks thickened and fibrotic, obscuring normal anatomic planes.
Key unsafe zones include the area directly adjacent to the common bile duct and hepatic artery, as aberrant anatomy or inflammation here increases the risk of vascular or biliary injury.
Safe dissection requires careful blunt dissection and possibly intraoperative cholangiography to delineate anatomy before proceeding.
The analysis is based on visual signs of inflammation and scarring.
The inflammation and scarring obscure key landmarks in Calot's triangle.
The obscured landmarks increase the risk of injuring the common bile duct and hepatic artery during laparoscopic cholecystectomy.


Format your output as a list of claims, one per line. Do not include any additional text. Be concise and avoid redundacy.

Now decompose the following paragraph into atomic, standalone, faithful cholecystectomy claims:
INPUT: {}
"""

decomposition_supernova = """
You will be given a paragraph that explains the reasoning behind classifying astrophysical time series data into one of the following categories: RR-Lyrae (RRL), peculiar type Ia supernova (SNIa-91bg), type Ia supernova (SNIa), superluminous supernova (SLSN-I), type II supernova (SNII), microlens-single (mu-Lens-Single), eclipsing binary (EB), M-dwarf, kilonova (KN), tidal disruption event (TDE), peculiar type Ia supernova (SNIax), type Ibc supernova (SNIbc), Mira variable, and active galactic nuclei (AGN). Your task is to decompose this explanation into individual claims that are:

Atomic: Each claim should express only one clear idea or judgment.
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph.
Faithful: The claims must preserve the original meaning and information.

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:

INPUT: The value data shows significant variability, with both positive and negative values, indicating the presence of a strong peak and subsequent decline, a hallmark of SNIa light curves.

OUTPUT:
The value data shows significant variability.
The data includes both positive and negative values.
The variability indicates the presence of a strong peak.
The data shows a subsequent decline after the peak.
A strong peak followed by decline is a hallmark of SNIa light curves.

Now decompose the following paragraph into atomic, standalone claims:
INPUT: {}
"""

decomposition_sepsis = """
You will be given a paragraph that explains the reasoning behind predicting whether a patient is at high risk of developing sepsis within the next 12 hours, based on electronic health record (EHR) data collected during the first 2 hours of their emergency department (ED) admission. Your task is to decompose this explanation into individual claims that are:

Atomic: Each claim should express only one clear idea or judgment.
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph.
Faithful: The claims must preserve the original meaning and information.

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:

INPUT: The respiratory rate is 22, which is on the higher side, and the pulse oximetry is 92%, indicating possible hypoxia.

OUTPUT:
The respiratory rate is 22.
A respiratory rate of 22 is on the higher side.
The pulse oximetry is 92%.
A pulse oximetry reading of 92% indicates possible hypoxia.

Now decompose the following paragraph into atomic, standalone claims:
INPUT: {}
"""

decomposition_emotion = """
You will be given a paragraph that explains why the given emotion was most reflected in a Reddit comment. Your task is to decompose this explanation into individual claims that are:

Atomic: Each claim should express only one clear idea or judgment.
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph.
Faithful: The claims must preserve the original meaning, nuance, and tone.

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:

INPUT: The use of "creepy" and "tbh" suggests a negative reaction or discomfort. Overall, this likely indicates the speaker finds the voicepack irritating or unsettling.

OUTPUT:
The use of "creepy" and "tbh" suggest a negative reaction.
The use of "creepy" and "tbh" suggests a feeling of discomfort.
The overall word choice likely indicates the speaker finds the voicepack irritating or unsettling.

Now decompose the following paragraph into atomic, standalone claims:
INPUT: {}
"""


decomposition_cardiac = """
You will be given a paragraph that explains the reasoning behind predicting whether a patient is at high risk of experiencing cardiac arrest within the next {}, based on basic patient background information (age, gender, race, reason for ICU admittance) and time-series Electrocardiogram (ECG) data at {} from the first {} of an ECG monitoring period during a patient's ICU stay. Your task is to decompose this explanation into individual claims that are:

Atomic: Each claim should express only one clear idea or judgment.
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph.
Faithful: The claims must preserve the original meaning and information.

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:

INPUT: The patient's ECG graph shows significant irregularities, with frequent and pronounced spikes and dips, indicating potential arrhythmic events. These patterns are concerning because they deviate from the normal consistent rhythm expected in a healthy heart. The spikes, particularly prominent around the 60 to 120-second marks, could signify ventricular tachycardia or fibrillation. Given that the patient is young, the primary risk factor stems from trauma-induced complications from the motor vehicle collision, such as cardiac tamponade or myocardial contusion, contributing to this prediction.

OUTPUT:
The patient's ECG graph shows significant irregularities with frequent and pronounced spikes and dips, indicating potential arrhythmic events.
The ECG patterns deviate from the normal consistent rhythm expected in a healthy heart.
The pronounced spikes on the ECG graph, particularly prominent around the 60 to 120-second marks, could signify ventricular tachycardia or fibrillation.
The patient is young, and the primary risk factor stems from trauma-induced complications from a motor vehicle collision.
Trauma-induced complications such as cardiac tamponade or myocardial contusion contribute to the prediction of high cardiac risk.

Now decompose the following paragraph into atomic, standalone claims:
INPUT: {}
"""

