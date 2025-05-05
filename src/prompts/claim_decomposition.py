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
Standalone: Each claim should be self-contained and understandable without needing to refer back to the paragraph.
Faithful: The claims must preserve the original meaning, nuance, and tone. Do not omit hedging language (e.g., "seems to," "somewhat," "lacks overt markers") or subjective phrasing if present.

Format your output as a list of claims separated by new lines. Do not include any additional text or explanations.

Here is an example of how to format your output:

INPUT: The weak lensing map shows a distribution of matter density with varying colors indicating different density levels. The presence of several yellow pixels suggests the existence of clusters, indicating regions of high matter density. These clusters are crucial for estimating Omega_m, as they reflect the total matter content in the universe. The blue areas represent voids, indicating low-density regions. The balance between these voids and clusters helps in estimating sigma_8, which measures the amplitude of matter fluctuations. The map shows a moderate number of clusters and voids, suggesting a balanced distribution of matter. This balance implies a moderate value for Omega_m, as there is neither an overwhelming presence of clusters nor voids. The presence of distinct clusters and voids also suggests a moderate value for sigma_8, indicating a typical level of matter fluctuation amplitude.

OUTPUT:
The weak lensing map shows a distribution of matter density with varying colors indicating different density levels.
The presence of several yellow pixels suggests the existence of clusters, indicating regions of high matter density.
The present clusters are crucial for estimating Omega_m, as they reflect the total matter content in the universe.
The blue areas on the map represent voids, indicating low-density regions.
The balance between voids and clusters on the map helps in estimating sigma_8, which measures the amplitude of matter fluctuations.
The map shows a moderate number of clusters and voids, suggesting a balanced distribution of matter.
A balanced distribution of matter implies a moderate value for Omega_m, as there is neither an overwhelming presence of clusters nor voids.
The presence of distinct clusters and voids suggests a moderate value for sigma_8, indicating a typical level of matter fluctuation amplitude.

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

INPUT:
1. Anatomical Landmarks and Safety: 
   - **Gallbladder**: Positioned to the right, the gallbladder is typically identified by its greenish hue. Safe regions for dissection include areas immediately adjacent to the gallbladder. 
   - **Liver**: Visible at the upper portion of the image, it must be preserved, thus avoiding direct dissection here is crucial.
   - **Cystic Duct and Artery**: These are critical structures that need careful identification and preservation to avoid bile leakage or bleeding. 

2. Specific Tissue Types:
   - **Peritoneum**: The thin, glistening tissue seen should be carefully cut to access deeper structures.
   - **Fibrous Tissue**: Dense fibrous connective tissue around the gallbladder may need to be severed to mobilize the gallbladder. Care should be taken here to avoid vital structures underneath.

3. Potential Risks and Complications:
   - **Bile Duct Injury**: Straying too medially can risk damage to the common bile duct, a serious complication.
   - **Bleeding**: Injury to the cystic or hepatic artery would lead to hemorrhage, requiring immediate control.


OUTPUT:
The gallbladder is typically identified by its greenish hue.
The gallbladder is positioned to the right.
The liver must be preserved, thus avoiding direct dissection here is crucial.
The cystic duct and artery need careful identification and preservation to avoid bile leakage or bleeding.
The peritoneum is the thin, glistening tissue seen should be carefully cut to access deeper structures.
The fibrous tissue around the gallbladder may need to be severed to mobilize the gallbladder.
Straying too medially can risk damage to the common bile duct, a serious complication.
Injury to the cystic or hepatic artery would lead to hemorrhage, requiring immediate control.


Format your output as a list of claims, one per line. Do not include any additional text. Be concise and avoid redundacy.

Now decompose the following paragraph into atomic, standalone, faithful cholecystectomy claims:
INPUT: {}
"""

decomposition_supernova = """
You will be given a paragraph that explains the reasoning behind classifying astrophysical time series data into one of the following categories: type Ia supernova (SNIa), type II supernova (SNII), M-dwarf, eclipsing binary (EB), tidal disruption event (TDE), type Ibc supernova (SNIbc), active galactic nuclei (AGN). Your task is to decompose this explanation into individual claims that are:

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
