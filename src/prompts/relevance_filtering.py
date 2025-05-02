relevance_template = """You will be given [description of input, output, and claim]

A claim is relevant if and only if:
(1) It is supported by the content of the input (i.e., it does not hallucinate or speculate beyond what is said).
(2) It helps explain why XXX.

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation of your judgment, pointing to specific support or lack thereof>

Here are some examples:

[Example 1]
[Example 2]
[Example 3]

Now, determine whether the following claim is relevant to the given XXX:
Input: {}
Output: {}
Claim: {}
"""

relevance_politeness = """You will be given an utterance, its politeness rating on a 1-5 scale (where 1: very rude and 5: very polite), and a claim that may or may not be relevant to an explanation of the rating. Your task is to decide whether the claim is relevant to explaining the politeness rating for this specific utterance.

A claim is relevant if and only if:
(1) It is supported by the content of the utterance (i.e., it does not hallucinate or speculate beyond what is said).
(2) It helps explain why the utterance received the given politeness rating (i.e., it directly relates to tone, phrasing, or other aspects relevant to the rating).

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation of your judgment, pointing to specific support or lack thereof>

Here are some examples:

[Example 1]
Utterance: "There is no such fact - you are just making things up. There is no reason to believe that any person reading about Bologna would be particularly interested in Kappa Sigma. If they wanted to know about Kappa Sigma, they would read the Kappa Sigma article instead.."
Politeness Rating: 2
Claim: The utterance accuses the other person of fabricating information.
Relevance: Yes
Reasoning: The claim is relevant because it discusses the accusatory tone of the utterance, which contributes to its rudeness classification.

[Example 2]
Utterance: "Deleted reference to REM sleep in the first sentence. It simply is not true. In fact, REM deprivation is a common side effect of antidepressant use (some attribute their effects to REM deprivation)."
Politeness Rating: 3
Claim: The utterance is neutral.
Relevance: No
Reasoning: The claim is not relevant because it simply states the rating, and does not provide information about why the rating was given. Claims that merely state the rating are not relevant.

[Example 3]
Utterance: "Tetra-gram is a compound word as is the penta-gram. Penta refers to the number 5 in Greek, tetra refers to the number 4 and gram refers to the word line in both cases. Obviously a star shape can't be shaped with 4 lines."
Politeness Rating: 3
Claim: The use of "obviously" might suggest the author is an expert in Greek.
Relevance: No
Reasoning: The claim is not relevant because there is nothing in the text to support that the author may be Greek. 

Now, determine whether the following claim is relevant to the given utterance and politeness rating:
Utterance: {}
Politeness Rating: {}
Claim: {}
"""


relevance_cholec = """
You will be given:
- An endoscopic image of the gallbladder region during a laparoscopic cholecystectomy (the "Input").
- A textual Claim describing a proposed "safe" or "unsafe" zone in that image (the "Claim").

A claim is relevant if and only if:
1. It refers to a visually detectable feature in the image.
2. It pertains to identifying safe or unsafe dissection zones based on expert surgical criteria.

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation pointing to the visual feature and criterion that supports your judgment>

Examples:

Example 1
Input: [image showing fully cleared Calot's triangle]
Claim: "Calot's triangle cleared."
```
Relevance: Yes
Reasoning: The claim describes the clearance of Calot's triangle, which is a visually confirmed safety criterion for safe dissection.
```

Example 2
Input: [image showing mottled gallbladder appearance]
Claim: "The gallbladder appears mottled green."
```
Relevance: No
Reasoning: While visible, the gallbladder color does not inform safe or unsafe dissection zones.
```

Now evaluate the following:
Input: (see attached image)
Claim: {}
"""

relevance_massmaps = """You will be given an image of a weak lensing mass map, its prediction for Omega_m and sigma_8, and a claim that may or may not be relevant to an explanation of the prediction. Your task is to decide whether the claim is relevant to explaining the prediction for this specific mass map.

A claim is relevant if and only if:
(1) It is supported by the content of the mass map (i.e., it does not hallucinate or speculate beyond what is said).
(2) It helps explain why the mass map received the given prediction (i.e., it directly relates to the mass map's features, such as the distribution of mass, the presence of voids or clusters, or the overall structure of the map).

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation of your judgment, pointing to specific support or lack thereof>

Here are some examples:

[Example 1]
Input: (Image 1)
Output: Omega_m = 0.1041, sigma_8 = 0.9396
Claim: The dataset represents the spatial distribution of matter density in the universe.
Relevance: No
Reasoning: This is a general statement and does not justify any specific prediction.

[Example 2]
Input: (Image 2)
Output: Omega_m = 0.3934, sigma_8 = 0.7018
Claim: The weak lensing map shows several yellow pixels close to each other on the left side, suggesting the existence of high-density regions or clusters.
Relevance: Yes
Reasoning: This is a specific cosmological structure observable in the data and indicative of cosmological parameters such as sigma_8.

[Example 3]
Input: (Image 3)
Output: Omega_m = 0.3586, sigma_8 = 0.9762
Claim: Voids are large low density regions in space.
Relevance: No
Reasoning: This is background knowledge, not derived from the data.

[Example 4]
Input: (Image 4)
Output: Omega_m = 0.4612, sigma_8 = 0.5614
Claim: There is a gray pixel in the upper left corner with value 6.2992e-04 in the data.
Relevance: No
Reasoning: Simply listing pixel values does not explain a prediction.

Now, determine whether the following claim is relevant to the given mass map and prediction:
Input: (Image 5)
Output: {}
Claim: {}
"""