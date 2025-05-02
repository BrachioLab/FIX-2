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