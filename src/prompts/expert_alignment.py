alignment_template = """You will be given [description of input, output, claim, and expert category]

Your task is as follows:
1. Determine which expert category is most aligned with the claim. 
2. Rate how strongly the category aligns with the claim on a scale of 0-1 (0 being lowest, 1 being highest. Use increments of 0.1). 

Return your answer as:
Category: <category>
Category Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

-----
Expert categories:
[list of categories and their descriptions]
-----

Here are some examples:
[Example 1]
[Example 2]
[Example 3]

Now, determine the category and alignment rating for the following claim:
Input: {}
Output: {}
Claim: {}
"""


alignment_poltieness = """You will be given an utterance, its politeness rating on a 1-5 scale (where 1: very rude and 5: very polite), and a claim that relates to why that rating was given. You will also be given a series of lexical categories that relate to politeness.

Your task is as follows:
1. Determine which lexical category is most aligned with the claim. 
2. Rate how strongly the category aligns with the claim on a scale of 0-1 (0 being lowest, 1 being highest). 

Return your answer as:
Category: <category>
Category Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

-----
Lexical categories:
1. Apologetic: Words and phrases used to acknowledge mistakes or express regret. (sorry, woops, oops, sry, apologize)
2. Deference: Polite words that convey respect, admiration, or acknowledgment of someone's status or authority. (great, good, nice, interesting, cool)
3. Direct Question: Words commonly used to form explicit questions seeking information or clarification. (what, where, why, who, when)
4. Discourse Marker: Common transition words used to structure speech or writing. (so, then, and, but, or)
5. Emergency: Phrases indicating urgency, immediate attention, or emergency situations. (right now, rn, as soon as possible, asap, immediately)
6. Factuality: Expressions that assert factual information or emphasize reality. (in fact, actually, the point, the reality, the truth)
7. First Person Plural: Sentences that contain a first-person plural pronoun. (we, our, ours, us, ourselves)
8. First Person Singular: Sentences that contain a first-person singular pronoun. (I, my, mine, myself, me)
9. First Person Start: Sentences that begin with a first-person singular pronoun. (I, my, mine, myself)
10. Gratitude: Words and phrases that express appreciation and thankfulness. (thanks, thx, thank you, thank u, i appreciate)
11. Greeting: Words and phrases used to initiate interaction or acknowledge someone’s presence. (hi, hello, hey)
12. Negative Emotion: Words and expressions that convey strong negative emotions or discontent. (bullshit, fuck, fucking, damn, shit)
13. Positive Emotion: Words and expressions that convey happiness, excitement, or approval. (abound, prefer, pride, priceless, pretty)
14. Hedging: Words that soften statements, making them less direct or assertive. (think, usually, unclearly, unclear, uncertainly)
15. Directive Speech Act: Expressions that instruct, command, or request an action from the listener. (can you, will you, can u, will u)
16. Indirectness: Words that introduce indirectness in communication, often for politeness or subtlety. (btw, by the way)
17. Ingroup Identity: Words that signal belonging to a specific social group or community. (mate, bro, homie, dude)
18. Politeness Marker: Words that make a request or instruction more courteous. (please, pls, plz, plse)
19. Polite Start: Sentences that begin with a politeness marker. (please, pls, plz)
20. Praise: Expressions that convey approval, admiration, or compliments. (awesome, outstanding, excellent, great, neat)
21. Commitment Marker: Words that express certainty or a strong commitment to an action or belief. (must, definitely, sure, definite, surely)
22. Second Person: Sentences that contain a second-person pronoun. (you, your, yours, yourself, u)
23. Second Person Start: Sentences that begin with a second-person pronoun. (you, your, yours, yourself)
24. Polite Request: Phrases that express politeness in requests or suggestions using modal verbs. (could you, would you, could u, would u)
25. Togetherness: Words that emphasize unity, collective action, or inclusivity. (together)
26. Direct Address: Words directly addressing the listener in conversation. (you, u)
-----

Here are some examples:
[Example 1]
Utterance: "There is no such fact - you are just making things up. There is no reason to believe that any person reading about Bologna would be particularly interested in Kappa Sigma. If they wanted to know about Kappa Sigma, they would read the Kappa Sigma article instead.."
Politeness Rating: 2
Claim: The utterance accuses the other person of fabricating information.
Category: Negative Emotion
Category Alignment Rating: 0.8
Reasoning: The accusatory tone and claim of fabrication imply a confrontational or hostile interaction, which strongly aligns with negative emotion. While the emotion is more implicit than explicit profanity or insult, the accusatory framing still carries a strong negative charge.

[Example 2]
Utterance: "Deleted reference to REM sleep in the first sentence. It simply is not true. In fact, REM deprivation is a common side effect of antidepressant use (some attribute their effects to REM deprivation)."
Politeness Rating: 3
Claim: The sentence structure is overly complex and difficult to follow.
Category: Discourse Marker
Category Alignment Rating: 0.1
Reasoning: The claim is about sentence complexity and structure, which does not relate to any of the listed lexical categories, including Discourse Marker, which refers to specific connecting words like “so” or “but.” The utterance does not exhibit structural markers that would directly contribute to complexity based on the provided categories, making the alignment very weak.

[Example 3]
Utterance: "Tetra-gram is a compound word as is the penta-gram. Penta refers to the number 5 in Greek, tetra refers to the number 4 and gram refers to the word line in both cases. Obviously a star shape can't be shaped with 4 lines."
Politeness Rating: 3
Claim: The use of "obviously" might suggest a slight assumption of common knowledge.
Category: Factuality
Category Alignment Rating: 0.6
Reasoning: The utterance presents factual information about word origins and geometric logic, and the claim focuses on the use of "obviously," which implies assumed knowledge rather than asserting a fact. This aligns somewhat with Factuality, but not strongly—Factuality is involved, but the assumption of common knowledge is more about tone than fact.

Now, determine the category and alignment rating for the following claim:
Utterance: {}
Politeness Rating: {}
Claim: {}
"""