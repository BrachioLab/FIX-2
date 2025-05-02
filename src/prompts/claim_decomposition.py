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


Format your output as a list of claims, one per line. Do not include any additional text.

Now decompose the following paragraph into atomic, standalone, faithful cholecystectomy claims:

INPUT: {}
"""