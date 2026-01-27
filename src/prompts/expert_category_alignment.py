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

Here are the definitions of the expert categories:
-----
1. Honorifics and Formal Address: The presence of respectful or formal address forms (e.g., “sir,” “usted,” “您”) signals politeness by expressing deference to the hearer’s status or social distance.
2. Courteous Politeness Markers: Words such as “please,” “kindly,” or their multilingual variants soften requests and reflect courteous intent.
3. Gratitude Expressions: Use of expressions like “thank you,” “thanks,” or “I appreciate it” signals recognition of the other’s contribution and positive face.
4. Apologies and Acknowledgment of Fault: Phrases such as “sorry” or “I apologize” express humility and repair social breaches, marking a clear politeness strategy.
5. Indirect and Modal Requests: Requests using modal verbs (“could you,” “would you”) or softening cues like “by the way” reduce imposition and signal respect for the hearer’s autonomy.
6. Hedging and Tentative Language: Words like “I think,” “maybe,” or “usually” lower assertion strength and make statements more negotiable, reflecting interpersonal sensitivity.
7. Inclusive Pronouns and Group-Oriented Phrasing: Use of “we,” “our,” or “together” expresses solidarity and reduces hierarchical distance in requests or critiques.
8. Greeting and Interaction Initiation: Opening with a salutation (“hi,” “hello”) creates a cooperative tone and frames the conversation positively.
9. Compliments and Praise: Positive evaluations (“great,” “awesome,” “neat”) attend to the hearer’s positive face and foster a friendly environment.
10. Softened Disagreement or Face-Saving Critique: When disagreeing, the use of softeners, partial agreements, or concern for clarity preserves the hearer’s dignity.
11. Urgency or Immediacy of Language: Utterances emphasizing emergency or speed (“asap,” “immediately”) can heighten perceived imposition and reduce politeness if not softened.
12. Avoidance of Profanity or Negative Emotion: The presence of strong negative words or swearing is a key indicator of rudeness and face threat.
13. Bluntness and Direct Commands: Requests lacking modal verbs or mitigation (“Do this”) are perceived as less polite due to their imperative structure.
14. Empathy or Emotional Support: Recognizing the hearer’s emotional context or challenges is a politeness strategy of concern and goodwill.
15. First-Person Subjectivity Markers: Statements that begin with “I think,” “I feel,” or “In my view” convey humility and subjectivity, reducing imposition.
16. Second Person Responsibility or Engagement: Sentences starting with “you” or directly addressing the hearer can either signal engagement or come across as accusatory, depending on context and tone.
17. Questions as Indirect Strategies: Questions (“what do you think?” or “could you clarify?”) reduce imposition by inviting rather than demanding input.
18. Discourse Management with Markers: Use of discourse markers like “so,” “then,” “but” organizes conversation flow and may help manage face needs in conflict or negotiation.
19. Ingroup Language and Informality: Use of group-identifying slang or casual expressions (“mate,” “dude,” “bro”) may foster solidarity or seem disrespectful, depending on relational norms.
-----

Return your answer as:
Reasoning: <A brief explanation of why you judged the alignment rating as you did.>
Category Alignment Rating: <rating>

Here are some examples:
[Example 1]
Category: Honorifics and Formal Address
Claims: ["The utterance uses the title "Dr." to address the other person.", "Use of "Dr." in the utterance is a sign of respect and politeness."]
Reasoning: The claims have complete alignment with the category because the claims directly talk about the use of a title to address the other person, which a type of honorific and formal address.
Category Alignment Rating: complete

[Example 2]
Category: Courteous Politeness Markers
Claims: ["The phrase "I'd bet money" implies disbelief in the other person's statement.", "Questioning the addressee's reading suggests they might be ignorant or careless.", "The statement is direct and disrespectful enough to be somewhat rude."]
Reasoning: The claims do not discuss the use of words such as "please", "kindly", or their multilingual variants to soften requests and reflect courteous intent, and therefore do not have any alignment with the category.
Category Alignment Rating: none

[Example 3]
Category: Avoidance of Profanity or Negative Emotion
Claims: ["The statement criticizes the other person's grammar in a condescending manner.", "The phrase "your grammar is terrible" is a direct insult.", "The commands about how to write English come across as patronizing.", "The speaker's tone is dismissive and unfriendly."]
Reasoning: Together, the claims relate to negativity and rudeness, which is related to profanity or negative emotion. However, the claims do not directly discuss the presence or lack of strong negative words or swearing, and therefore do not have complete alignment with the category.
Category Alignment Rating: partial

Now, determine the alignment rating for the following expert category and set of claims:
Category: {}
Claims: {}
"""

category_alignment_emotion = """You will be given a set of claims that relate to why an emotion label was given to a piece of text, and a series of categories that an expert  psychologist would use to perform this type of emotion classification.

Your task is as follows:
Rate how strongly the set of claims align with the category. Choose from complete, partial, or none.

Alignment explanations:
Complete: The claim is specific, directly relevant, and fully captures the meaning and intent of the expert category.
Partial: The claim partially refers to the expert category but lacks key details, uses vague language, is overly general, or contains noise.
None: The claim references something unrelated to the expert category, or misinterprets the category's meaning.

Here are the definitions of the expert categories:
-----
1. Valence: Decide if the overall tone is pleasant or unpleasant; positive tones suggest joy or admiration, negative tones suggest sadness or anger.
2. Arousal: Gauge how energized the wording is—calm phrasing implies low arousal emotions, intense phrasing implies high arousal emotions.
3. Emotion Words & Emojis: Look for direct emotion terms or emoticons that explicitly name the feeling.
4. Expressive Punctuation: Multiple exclamation marks, ALL-CAPS, or stretched spellings signal higher emotional intensity.
5. Humor/Laughter Markers: Tokens like “haha,” “lol,” or laughing emojis reliably indicate amusement.
6. Confusion Phrases: Statements such as “I don’t get it” clearly mark confusion.
7. Curiosity Questions: Genuine information-seeking phrases (“I wonder…”, “why is…?”) point to curiosity.
8. Surprise Exclamations: Reactions of astonishment (“No way!”, “I can’t believe it!”) denote surprise.
9. Threat/Worry Language: References to danger or fear (“I’m scared,” “terrifying”) signal fear or nervousness.
10. Loss or Let-Down Words: Mentions of loss or disappointment cue sadness, disappointment, or grief.
11. Other-Blame Statements: Assigning fault to someone else for a bad outcome suggests anger or disapproval.
12. Self-Blame & Apologies: Admitting fault and saying “I’m sorry” marks remorse.
13. Aversion Terms: Words like “gross,” “nasty,” or “disgusting” point to disgust.
14. Praise & Compliments: Positive evaluations of someone’s actions show admiration or approval.
15. Gratitude Expressions: Phrases such as “thanks” or “much appreciated” indicate gratitude.
16. Affection & Care Words: Loving or nurturing language (“love this,” “sending hugs”) signals love or caring.
17. Self-Credit Statements: Boasting about one’s own success (“I nailed it”) signals pride.
18. Relief Indicators: Release phrases like “phew,” “finally over,” or “what a relief” mark relief after stress ends.
-----

Return your answer as:
Reasoning: <A brief explanation of why you judged the alignment rating as you did.>
Category Alignment Rating: <rating>

Here are some examples:
[Example 1]
Category: Humor/Laughter Markers
Claims: ["The text contains laughter ("hahaha").", "Laughter is an indicator of amusement.", "The text contains an expression of surprise and lightheartedness ("omg").", ""omg" is an indicator of amusement.", "The phrase "sling yoghurt" is mentioned in a playful context.", "The playful context contributes to the humorous tone.", "The overall expression is informal.", "The overall expression is casual."]
Reasoning: The claims explicitly mention laughter and connect it to amusement, which is exactly what humor/laughter markers capture. Additional claims about playful and humorous tone support the same signal. The informal/casual style adds context but does not detract from the clear laughter evidence. Overall, the set strongly and directly aligns with humor markers.
Category Alignment Rating: complete

[Example 2]
Category: Arousal
Claims: ["The text conveys a strong negative judgment towards the subject's perceived hypocrisy.", "Words such as "perfectly comfortable" paired with "hypocrite" suggest criticism.", "The phrase "utterly outraged" emphasizes the speaker's disapproval of the subject's behavior."]
Reasoning: The phrase “utterly outraged” suggests heightened intensity, which relates to arousal. However, most claims focus on judgment, criticism, and disapproval rather than the energy level of the wording. Because only a subset of claims directly signal intensity, the alignment is partial.
Category Alignment Rating: partial

[Example 3]
Category: Expressive Punctuation
Claims: ["The tone of defensiveness aligns with feelings of annoyance.", "The comment suggests a negative tension."]
Reasoning: Expressive punctuation looks for exclamation marks, ALL-CAPS, or stretched spellings. None of the claims mention punctuation or capitalization patterns, so there is no alignment with this category.
Category Alignment Rating: none

Now, determine the alignment rating for the following expert category and set of claims:
Category: {}
Claims: {}
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

Example 2:
INPUT:
Category: Early Antibiotic/Culture Orders (within 2 hours): Administration of broad‑spectrum antibiotics or drawing of blood cultures within the first 2 hours signifies clinician suspicion of serious infection and should anchor sepsis risk assessment.
Claims:
The patient exhibits several risk factors for sepsis.

OUTPUT:
Reasoning: The claim is a general statement about sepsis risk and provides no evidence of antibiotic administration, blood culture orders, or early clinical intervention. As such, it does not support this category, resulting in no alignment.
Category Alignment Rating: none

Example 3:
INPUT:
Category: SIRS Positivity (≥2 Criteria): Presence of ≥ 2 SIRS criteria—temperature > 38 °C or < 36 °C, heart rate > 90 bpm, respiratory rate > 20 /min or PaCO₂ < 32 mm Hg, or WBC > 12 000/µL or < 4 000/µL—identifies systemic inflammation consistent with early sepsis.
Claims:
Another risk factor for sepsis is a high triage temperature.
A high triage temperature indicates fever.
The respiratory rate is 26.
Another risk factor for sepsis is an elevated respiratory rate.

OUTPUT:
Reasoning: The claims collectively establish the presence of two SIRS criteria. A high triage temperature is explicitly identified and interpreted as fever, satisfying the SIRS temperature criterion (> 38 °C). In addition, the respiratory rate of 26/min exceeds the SIRS threshold (> 20/min), and is explicitly noted as an elevated respiratory rate. Together, these findings meet ≥ 2 SIRS criteria, indicating systemic inflammation consistent with early sepsis.
Category Alignment Rating: complete

Example 4:
INPUT:
Category: High qSOFA Score (≥2): A qSOFA score ≥ 2 (respiratory rate ≥ 22 /min, systolic BP ≤ 100 mmHg, or altered mentation) flags high risk of sepsis‑related organ dysfunction and mortality.
Claims:
The respiratory rate is 26.
A respiratory rate of 26 is concerning for sepsis.

OUTPUT:
Reasoning: The claims support only one qSOFA criterion: a respiratory rate of 26/min, which exceeds the qSOFA threshold (≥ 22/min). However, there is no evidence provided for the other qSOFA components—systolic blood pressure ≤ 100 mmHg or altered mentation. Since a qSOFA score of ≥ 2 is required and only one criterion is satisfied, the alignment with the category is partial.
Category Alignment Rating: partial

Example 5:
INPUT:
Category: Sepsis-Associated Hypotension (SBP <90 mmHg or MAP <70 mmHg, or ≥40 mmHg drop): Sepsis‑associated hypotension, defined as SBP < 90 mmHg, MAP < 70 mmHg, or a ≥ 40 mmHg drop from baseline, indicates progression toward septic shock.
Claims:
The patient’s systolic blood pressure is 85 mmHg.
Mean arterial pressure is 65 mmHg.
Vasopressor support was initiated due to low blood pressure.

OUTPUT:
Reasoning: The claims directly satisfy the definition of sepsis-associated hypotension. A systolic blood pressure of 85 mmHg is below the SBP threshold (< 90 mmHg), and a mean arterial pressure of 65 mmHg is below the MAP threshold (< 70 mmHg). The initiation of vasopressor support due to low blood pressure further corroborates clinically significant hypotension consistent with progression toward septic shock. Together, these findings strongly and unambiguously align with the category.
Category Alignment Rating: complete

Now, determine the alignment rating for the following expert category and set of claims:
Category: {}
Claims: {}
"""

category_alignment_supernova = """You will be given a set of claims explaining why a this time series dataset is classified as such astroph ysical class.
multivariate time series visualized as a scatter plot image. The x-axis represents time, and the y-axis represents the flux measurement value. Each point corresponds to an observation at a specific timestamp and wavelength. Different wavelengths are color-coded, and observational uncertainty is shown using vertical error bars.
patient was predicted to be at high or low risk of sepsis within the next 12 hours (Yes/No). You will also be given a series of categories that an expert clinician would use to perform this type of sepsis prediction.

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
Category: Contiguous non-zero flux: Contiguous non‑zero flux segments confirm genuine astrophysical activity and define the time windows from which transient features should be extracted.
Claims:
The time series shows a rapid rise to a peak with subsequent decline.
A rapid rise to a peak with subsequent decline is characteristic of a Type Ia supernova light curve.

OUTPUT:
Reasoning: The claims directly describe a continuous, coherent flux evolution with a clear rise and decline, which indicates sustained non-zero flux over a contiguous time window. This exactly matches the intent of the category, which focuses on identifying genuine astrophysical activity and defining valid transient segments.
Category Alignment Rating: complete

Example 2:
INPUT:
Category: Rise–decline rates: Characteristic rise‑and‑decline rates—such as the fast‑rise/slow‑fade morphology of many supernovae—encode energy‑release physics and serve as strong class discriminators.
Claims:
The time series shows a rapid increase in brightness followed by a gradual decline.
The time series includes an initial flat phase followed by a sharp increase and consistent decline.
This rise and decline pattern is characteristic of type Ia supernovae.

OUTPUT:
Reasoning: The claims explicitly describe the speed and shape of the brightness evolution, including a rapid rise and a slower, consistent decline, which are the defining elements of rise–decline rates. They also connect this morphology to a specific supernova class, fully capturing both the physical interpretation and discriminative purpose of the category.
Category Alignment Rating: complete

Example 3:
INPUT:
Category: Monotonic flux trends: Locally smooth, monotonic flux trends across one or multiple bands (plateaus, linear decays) capture physical evolution stages and help distinguish SN II‑P, SN II‑L, and related classes.
Claims:
An initial flat phase followed by a sharp increase and decline.

OUTPUT:
Reasoning: The claim mentions an initial flat phase, which aligns with the idea of a monotonic plateau capturing a physical evolution stage. However, the subsequent sharp increase and decline introduce non-monotonic behavior and do not clearly describe a smooth, monotonic trend across a band, nor do they distinguish between SN II-P and SN II-L. As a result, the claim only partially captures the intent of the category.
Category Alignment Rating: partial

Example 4:
INPUT:
Category: Secondary maxima: Filter‑specific secondary maxima or shoulders in red/near‑IR bands—prominent in SNe Ia—are morphological features absent in most core‑collapse SNe.
Claims:
N/A

OUTPUT:
Reasoning: There are no claims provided and therefore no information that could align with the expert category.
Category Alignment Rating: none

Example 5:
INPUT:
Category: Event duration: Total event duration, measured from first detection to return to baseline, distinguishes short‑lived kilonovae and superluminous SNe from longer plateau or AGN variability phases.
Claims:
The flux variability pattern is persistent over a long duration.
Irregular and persistent flux variability is characteristic of active galactic nuclei (AGN).

OUTPUT:
Reasoning: Both claims directly address long-lasting variability, which is the core concept of event duration. The first explicitly states that the activity persists over a long time, and the second links this long-duration, irregular behavior to AGN, a class distinguished from short-lived transients precisely by extended event duration.
Category Alignment Rating: complete

Now, determine the alignment rating for the following expert category and set of claims:
Category: {}
Claims: {}
"""