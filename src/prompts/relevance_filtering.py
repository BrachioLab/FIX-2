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
"""
