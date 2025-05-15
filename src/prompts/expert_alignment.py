alignment_template = """You will be given [description of claim and expert category]

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
Claim: {}
"""


alignment_politeness = """You will be given a single claim that relates to why a politeness rating was given to an utterance. You will also be given a series of categories that an expert linguist would use to perform this type of politeness classification.

Your task is as follows:
1. Determine which expert category is most aligned with the claim. 
2. Rate how strongly the category aligns with the claim on a scale of 0-1 (0 being lowest, 1 being highest. Use increments of 0.1). 

Return your answer as:
Category: <category>
Category Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

-----
Expert politeness categories:
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

Here are some examples:
[Example 1]
Claim: The utterance accuses the other person of fabricating information.
Category: Second Person Responsibility or Engagement
Category Alignment Rating: 0.9
Reasoning: The claim centers on the utterance accusing the other person of fabricating information, which aligns strongly with the use of direct second-person language (“you are just making things up”). This construction directly targets the listener and assigns blame, which is a key aspect of the Second Person Responsibility or Engagement category. 

[Example 2]
Claim: The sentence structure is overly complex and difficult to follow.
Category: Discourse Management with Markers
Category Alignment Rating: 0.1
Reasoning: The claim concerns sentence complexity and readability, which is weakly aligned with how Discourse Management with Markers functions—lack of these markers impact information organization or textual flow. However, this category has low alignment because the claim is more about syntactic complexity than interpersonal politeness strategies. No other category more directly addresses sentence structure, so this is the closest fit, but with very low alignment.

[Example 3]
Claim: The use of "obviously" might suggest a slight assumption of common knowledge.
Category: Hedging and Tentative Language
Category Alignment Rating: 0.6
Reasoning: The word "obviously" implies a presumption that the information should be universally known or accepted, which can reduce the speaker's humility and increase the assertiveness of the statement. This contrasts with hedging strategies that usually aim to soften claims and maintain interpersonal sensitivity. While "obviously" itself is not a hedge, its use relates to the degree of assertiveness in an utterance—making this category moderately relevant. The alignment is not perfect because “obviously” actually opposes hedging, but the connection lies in the shared focus on epistemic stance and assertion strength.

Now, determine the category and alignment rating for the following claim:
Claim: {}
"""


alignment_cholec = """You will be given:
- A textual Claim describing a proposed "safe" or "unsafe" zone in laparoscopic cholecystectomy (the "Claim").

Your task is as follows:
1. Determine which expert safety criterion is most aligned with the Claim.
2. Rate how strongly the criterion aligns with the Claim on a scale of 0.0-1.0 (in increments of 0.1), where 0.0 means "not at all" and 1.0 means "perfectly."

Return your answer exactly in this format:
Category: <name of selected criterion>,
Category ID: <the ID of the selected criterion>
Alignment: <the alignment rating for this category>
Reasoning: <A brief explanation of why you selected the criterion and how you judged the alignment rating>

-----
Expert Safety Criteria:
1. Calot's triangle cleared - Hepatocystic triangle must be fully cleared of fat/fibrosis so that its boundaries are unmistakable.
2. Cystic plate exposed - The lower third of the gallbladder must be dissected off the liver to reveal the shiny cystic plate and ensure the correct dissection plane.
3. Only two structures visible - Only the cystic duct and cystic artery should be seen entering the gallbladder before any clipping or cutting.
4. Above the R4U line - Dissection must remain cephalad to an imaginary line from Rouviere's sulcus to liver segment IV to avoid the common bile duct.
5. Safe distance from common bile duct - There should be sufficient distance between the common bile duct and the gallbladder wall to ensure safe dissection.
6. Infundibulum start point - Dissection should begin at the gallbladder infundibulum-cystic duct junction to stay in safe tissue planes.
7. Subserosal plane stay - When separating the gallbladder from the liver, stay in the avascular subserosal cleavage plane under the serosal fat layer.
8. Cystic lymph node guide - Identify the cystic lymph node and clip the artery on the gallbladder side of the node to avoid injuring the hepatic artery.
9. No division without ID - Never divide any duct or vessel until it is unequivocally identified as the cystic structure entering the gallbladder.
10. Inflammation bailout - If dense scarring or distorted anatomy obscures Calot's triangle, convert to a subtotal "fundus-first" approach rather than blind cutting.
11. Aberrant artery caution - Preserve any large or tortuous artery (e.g., a Moynihan's hump) that might be mistaken for the cystic artery.

-----
Examples:

Example 1
Claim: "The fat and fibrous tissue overlying Calot's triangle has been fully excised, exposing only two tubular structures."

Output 1:
Category: Calot's triangle cleared
Category ID: 1
Alignment: 1.0
Reasoning: The claim precisely describes complete clearance of Calot's triangle, matching this criterion perfectly.


Example 2
Claim: "The cystic plate is not visible due to dense adhesions, making the gallbladder-liver plane indistinct."

Output 2:
Category: Cystic plate exposed
Category ID: 2
Alignment: 0.2
Reasoning: The claim refers to difficulty visualizing the cystic plate because of adhesions, which relates to this criterion but indicates failure, hence a low score.


Now, determine the category and alignment rating for the following claim:
Claim: [[CLAIM]]
"""

alignment_massmaps = """You will be given a single claim that relates to why a prediction was given to a mass map. You will also be given a series of categories that an expert cosmologist would use to perform this type of cosmological parameter prediction.

Your task is as follows:
1. Determine which expert category is most aligned with the claim. 
2. Rate how strongly the category aligns with the claim on a scale of 0-1 (0 being lowest, 1 being highest. Use increments of 0.1). 

Return your answer as:
Category: <category>
Category ID: <category ID>
Category Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

-----
Expert cosmology categories:
1. Lensing Peak (Cluster) Abundance: A higher count of prominent, high-convergence peaks in the map indicates a larger sigma_8, since a clumpier matter distribution produces more frequent massive halos.
2. Void Size and Frequency: Extensive low-convergence void regions suggest a lower Omega_m, as a reduced overall matter density allows bigger underdense expanses to form in the cosmic web.
3. Filament Thickness and Sharpness: Bold, sharply defined filaments threading between clusters imply a higher sigma_8 (stronger small-scale clustering), whereas thin or diffuse filaments point to a lower amplitude of matter fluctuations.
4. Fine-Scale Clumpiness: A grainy, fine-textured pattern of small-scale lensing fluctuations (many mini-clumps) is a visual signature of high sigma_8, whereas a smoother, more homogeneous map suggests a lower sigma_8.
5. Connectivity of the Cosmic Web: A highly interconnected filament network (with filaments linking most clusters into a continuous web) hints at a higher Omega_m, whereas a more fragmented scene of isolated clumps separated by wide gaps is expected for a lower Omega_m.
6. Density Contrast Extremes: Very pronounced contrast between dense regions and empty voids - i.e. bright lensing peaks adjacent to dark void areas - signals an enhanced variance of the density field (high sigma_8), whereas subdued contrast suggests lower sigma_8.
-----

Here are some examples:
Example 1
Claim: "There exist a large amount of yellow regions in the map, which indicates a relatively high sigma_8."

Output 1:
Category: Lensing Peak (Cluster) Abundance
Category ID: 1
Alignment: 1.0
Reasoning: The claim directly talks about a large number of yellow regions (high-convergence peaks) in the map, and how it indicates high sigma_8. This aligns with the Lensing Peak (Cluster) Abundance category which says a large number of peaks / clusters indicates a larger sigma_8.

Example 2
Claim: "The presence of some void regions in the map indicates a low matter density level."

Output 2:
Category: Void Size and Frequency
Category ID: 2
Alignment: 0.5
Reasoning: The claim does mention that void should lead to a lower matter density level and thus Omega_m, but it does not mention how large the void is. To be completely aligned with the expert criteria, the claim should mention the size of the void and it should be large to match this category.

Example 3
Claim: "A balanced distribution of blue, gray, red, and yellow regions in the map indicates a moderate matter density level and fluctuation levels."

Output 3:
Category: Density Contrast Extremes
Category ID: 6
Alignment: 0.1
Reasoning: There is no category saying that when the distribution is balanced, whether the matter density level should be high or low or moderate. This is the closest category because it mentions both voids and peaks, but the alignment is only 0.1 because the expert criteria does not mention balanced distribution but rather the position of the peaks and voids.

Now, determine the category and alignment rating for the following claim:
Claim: [[CLAIM]]
"""

alignment_massmaps_mapping = {
    'name2id': {
        'Lensing Peak (Cluster) Abundance': 1,
        'Void Size and Frequency': 2,
        'Filament Thickness and Sharpness': 3,
        'Fine-Scale Clumpiness': 4,
        'Connectivity of the Cosmic Web': 5,
        'Density Contrast Extremes': 6,
    },
    'id2name': {
        1: 'Lensing Peak (Cluster) Abundance',
        2: 'Void Size and Frequency',
        3: 'Filament Thickness and Sharpness',
        4: 'Fine-Scale Clumpiness',
        5: 'Connectivity of the Cosmic Web',
        6: 'Density Contrast Extremes',
    }
}

alignment_emotion = """You will be given a single claim that relates to why an emotion label was assigned to a piece of text. You will also be given a series of categories that an expert emotion psychologist would use to perform this type of emotion classification.

Your task is as follows:
1. Determine which expert category is most aligned with the claim. 
2. Rate how strongly the category aligns with the claim on a scale of 0-1 (0 being lowest, 1 being highest). 

Return your answer as:
Category: <category>
Category Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

-----
Expert emotion categories:
1. Valence: Decide if the overall tone is pleasant or unpleasant; positive tones suggest joy or admiration, negative tones suggest sadness or anger.
2. Arousal: Gauge how energized the wording is—calm phrasing implies low arousal emotions, intense phrasing implies high arousal emotions.
3. Emotion Words & Emojis: Look for direct emotion terms or emoticons that explicitly name the feeling.
4. Expressive Punctuation: Multiple exclamation marks, ALL‑CAPS, or stretched spellings signal higher emotional intensity.
5. Humor/Laughter Markers: Tokens like “haha,” “lol,” or laughing emojis reliably indicate amusement.
6. Confusion Phrases: Statements such as “I don’t get it” clearly mark confusion.
7. Curiosity Questions: Genuine information‑seeking phrases (“I wonder…”, “why is…?”) point to curiosity.
8. Surprise Exclamations: Reactions of astonishment (“No way!”, “I can’t believe it!”) denote surprise.
9. Threat/Worry Language: References to danger or fear (“I’m scared,” “terrifying”) signal fear or nervousness.
10. Loss or Let‑Down Words: Mentions of loss or disappointment cue sadness, disappointment, or grief.
11. Other‑Blame Statements: Assigning fault to someone else for a bad outcome suggests anger or disapproval.
12. Self‑Blame & Apologies: Admitting fault and saying “I’m sorry” marks remorse.
13. Aversion Terms: Words like “gross,” “nasty,” or “disgusting” point to disgust.
14. Praise & Compliments: Positive evaluations of someone’s actions show admiration or approval.
15. Gratitude Expressions: Phrases such as “thanks” or “much appreciated” indicate gratitude.
16. Affection & Care Words: Loving or nurturing language (“love this,” “sending hugs”) signals love or caring.
17. Self‑Credit Statements: Boasting about one’s own success (“I nailed it”) signals pride.
18. Relief Indicators: Release phrases like “phew,” “finally over,” or “what a relief” mark relief after stress ends.
-----

Here are some examples:
[Example 1]
Claim: The exclamations (“seriously wtf… sickest soulread ever”) show astonished praise for an impressive play.
Category: Surprise Exclamations
Category Alignment Rating: 0.7
Reasoning: The phrase “seriously wtf… sickest soulread ever” indicates a strong emotional reaction of astonishment. The use of “seriously” as an intensifier and the informal hyperbolic phrasing align closely with surprise. The reason the rating is 0.7 and not higher is that while there is a strong element of astonishment, the claim also suggests admiration or praise, which is more related to Praise & Compliments. Though surprise is the more dominant signal, the claim does not fully relate to surprise.

[Example 2]
Claim: The text discusses economic inflation trends over the past decade.
Category: Emotion Words & Emojis
Category Alignment Rating: 0.0
Reasoning: This claim describes factual, analytical content about economic trends and contains no reference to emotional language, emojis, or affective tone. Since the Emotion Words & Emoji category looks for direct markers of emotion (like “happy,” “sad,” “😭”), and none are present or implied in the claim, the alignment is essentially nonexistent. 

[Example 3]
Claim: The speaker is expressing happiness at a positive outcome.
Category: Valence
Category Alignment Rating: 1.0
Reasoning: The claim directly identifies the emotional tone as happiness and attributes it to a positive outcome, which maps precisely onto the Valence category. Valence is concerned with whether the emotional tone is pleasant or unpleasant, and happiness due to a good result is a prototypical example of high positive valence. There are no additional cues in the claim suggesting other categories (like specific emotion words or punctuation), so Valence is both the most relevant and strongly aligned.

Now, determine the category and alignment rating for the following claim:
Claim: {}
"""

alignment_supernova = """You will be given a single claim that relates to why time-series data from a supernova was classified as a certain supernova class. The classes are as follows: RR-Lyrae (RRL), peculiar type Ia supernova (SNIa-91bg), type Ia supernova (SNIa), superluminous supernova (SLSN-I), type II supernova (SNII), microlens-single (mu-Lens-Single), eclipsing binary (EB), M-dwarf, kilonova (KN), tidal disruption event (TDE), peculiar type Ia supernova (SNIax), type Ibc supernova (SNIbc), Mira variable, and active galactic nuclei (AGN). You will also be given a series of categories that an expert astrophysicist would use to perform this type of supernova classification from time-series data.
Your task is as follows:
1. Determine which expert category is most aligned with the claim. 
2. Rate how strongly the category aligns with the claim on a scale of 0-1 (0 being lowest, 1 being highest). 

Return your answer as:
Category: <category>
Category Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

-----
Expert astrophysical categories:
1. Contiguous non‑zero flux segments confirm genuine astrophysical activity and define the time windows from which transient features should be extracted.
2. Characteristic rise‑and‑decline rates—such as the fast‑rise/slow‑fade morphology of many supernovae—encode energy‑release physics and serve as strong class discriminators.
3. Peak‑to‑trough photometric amplitude separates high‑energy explosive events (multi‑magnitude outbursts) from low‑amplitude periodic or stochastic variables.
4. Total event duration, measured from first detection to return to baseline, distinguishes short‑lived kilonovae and superluminous SNe from longer plateau or AGN variability phases.
5. Periodic light curves with stable periods and distinctive Fourier amplitude‑ and phase‑ratios (e.g., φ21, φ31) flag pulsators and eclipsing binaries rather than one‑off transients.
6. Filter‑specific secondary maxima or shoulders in red/near‑IR bands—prominent in SNe Ia—are morphological features absent in most core‑collapse SNe.
7. Locally smooth, monotonic flux trends across one or multiple bands (plateaus, linear decays) capture physical evolution stages and help distinguish SN II‑P, SN II‑L, and related classes.

-----

Here are some examples:
[Example 1]
Claim: Observations are recorded at various wavelength overtime.
Category: Locally smooth, monotonic flux trends across one or multiple bands (plateaus, linear decays) capture physical evolution stages and help distinguish SN II‑P, SN II‑L, and related classes.
Category Alignment Rating: 
Score: 0
Reasoning: This is a general data format description rather than a discriminative feature. While it relates tangentially to Multi-wavelength Observability, it lacks any mention of spectral variation or joint analysis, and doesn't aid in distinguishing classes.

[Example 2]
Claim: Consistent and distinct peaks are observed in value data at specific wavelength.
Category: Periodic light curves with stable periods and distinctive Fourier amplitude‑ and phase‑ratios (e.g., φ21, φ31) flag pulsators and eclipsing binaries rather than one‑off transients.
Category Alignment Rating: 0.6
Reasoning: The mention of consistent and distinct peaks hints at periodic behavior, potentially corresponding to pulsators or binaries. However, the claim doesn’t mention periods or Fourier components explicitly, so the match is partial.

[Example 3]
Claim: Variations in intensity over time are typical of the lightcurve evolution of a supernova.
Category: Locally smooth, monotonic flux trends across one or multiple bands (plateaus, linear decays) capture physical evolution stages and help distinguish SN II‑P, SN II‑L, and related classes.
Category Alignment Rating: 0.7
Reasoning: The claim refers to general light curve behavior, which aligns with the concept of smooth, evolving flux. It supports class differentiation but is too vague to tie to a specific morphological trend or SN type, hence not a perfect match.

[Example 4]
Claim: The flux value has a rapid increase and gradual decrease.
Category: Characteristic rise‑and‑decline rates—such as the fast‑rise/slow‑fade morphology of many supernovae—encode energy‑release physics and serve as strong class discriminators.
Category Alignment Rating: 1.0
Reasoning: This is a direct description of the fast-rise/slow-decline morphology mentioned in the expert criteria. It is a classic feature of many supernovae and maps perfectly to this category.

[Example 5]
Claim: Significant fluctuations and peaks in the data can be inferred as part of the light curve of a Type II supernova.
Category: Locally smooth, monotonic flux trends across one or multiple bands (plateaus, linear decays) capture physical evolution stages and help distinguish SN II‑P, SN II‑L, and related classes.
Category Alignment Rating: 0.8
Reasoning: Type II SNe, especially SN II-P and SN II-L, are characterized by distinct plateau or linear decline phases. The claim’s wording aligns with the idea of significant, structured flux evolution but lacks specifics (e.g., plateau shape), so the alignment is strong but not perfect.

Now, determine the category and alignment rating for the following claim:
Claim: {}
"""

alignment_sepsis = """You will be given a single claim explaining why a patient was predicted to be at high or low risk of sepsis within the next 12 hours (Yes/No). You will also be given a series of categories that an expert clinician would use to perform this type of sepsis prediction.

Your task is as follows:
1. Determine which expert category is most aligned with the claim. 
2. Rate how strongly the category aligns with the claim on a scale of 0-1 (0 being lowest, 1 being highest. Use increments of 0.1). 

Return your answer as:
Category: <category>
Category Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

-----
Expert sepsis categories:
1. Advanced age (≥ 65 years) markedly increases susceptibility to rapid sepsis progression and higher mortality after infection.
2. Presence of ≥ 2 SIRS criteria—temperature > 38 °C or < 36 °C, heart rate > 90 bpm, respiratory rate > 20 /min or PaCO₂ < 32 mm Hg, or WBC > 12 000/µL or < 4 000/µL—identifies systemic inflammation consistent with early sepsis.
3. A qSOFA score ≥ 2 (respiratory rate ≥ 22 /min, systolic BP ≤ 100 mmHg, or altered mentation) flags high risk of sepsis‑related organ dysfunction and mortality.
4. A National Early Warning Score (NEWS) of ≥ 5–7 derived from deranged vitals predicts imminent clinical deterioration compatible with sepsis.
5. Serum lactate ≥ 2 mmol/L within the first 2 hours signals tissue hypoperfusion and markedly elevates sepsis mortality risk.
6. Shock index (heart rate ÷ systolic BP) ≥ 1.0—or a rise ≥ 0.3 from baseline—denotes haemodynamic instability and a high probability of severe sepsis.
7. Sepsis‑associated hypotension, defined as SBP < 90 mmHg, MAP < 70 mmHg, or a ≥ 40 mmHg drop from baseline, indicates progression toward septic shock.
8. An increase of ≥ 2 points in any SOFA component—e.g., PaO₂/FiO₂ < 300, platelets < 100 × 10⁹/L, bilirubin > 2 mg/dL, creatinine > 2 mg/dL, or GCS < 12—confirms new organ dysfunction and high sepsis risk.
9. Administration of broad‑spectrum antibiotics or drawing of blood cultures within the first 2 hours signifies clinician suspicion of serious infection and should anchor sepsis risk assessment.

-----

Here are some examples:
[Example 1]
Claim: The patient is 86 years old. 
Category: Advanced age (≥ 65 years) markedly increases susceptibility to rapid sepsis progression and higher mortality after infection.
Category Alignment Rating: 1.0
Reasoning: The claim directly references advanced age (≥ 65), which this category identifies as a strong sepsis risk factor—perfect alignment.

[Example 2]
Claim: The patient's temperature is 99.6°F, close to the fever threshold.
Category: Presence of ≥ 2 SIRS criteria—temperature > 38 °C or < 36 °C, heart rate > 90 bpm, respiratory rate > 20 /min or PaCO₂ < 32 mm Hg, or WBC > 12 000/µL or < 4 000/µL—identifies systemic inflammation consistent with early sepsis.
Category Alignment Rating: 0.4
Reasoning: The temperature is close to the SIRS threshold (>38°C), which makes it marginally relevant. It hints at possible early sepsis, but does not meet the SIRS criterion.

[Example 3]
Claim: The patient's pulse oximetry is 92%, indicating possible hypoxia.
Category: An increase of ≥ 2 points in any SOFA component—e.g., PaO₂/FiO₂ < 300, platelets < 100 × 10⁹/L, bilirubin > 2 mg/dL, creatinine > 2 mg/dL, or GCS < 12—confirms new organ dysfunction and high sepsis risk.
Category Alignment Rating: 0.6
Reasoning: Pulse oximetry isn’t directly a SOFA measure, but hypoxia relates conceptually to impaired oxygenation (PaO₂/FiO₂). The alignment is indirect but meaningful.

[Example 4]
Claim: An elevated white blood cell count suggests an inflammatory or infectious process.
Category: Presence of ≥ 2 SIRS criteria
Category Alignment Rating: 1.0
Reasoning: Elevated WBC is explicitly listed under SIRS criteria, which are used to identify systemic inflammation in early sepsis.

[Example 5]
Claim: The patient's condition warrants close monitoring and further investigation for infection.
Category: Administration of broad‑spectrum antibiotics or drawing of blood cultures…
Category Alignment Rating: 0.5
Reasoning: While this reflects suspicion of infection, it lacks concrete clinical action (like antibiotic administration or culture draws). 

Now, determine the category and alignment rating for the following claim:
Claim: {}
"""


alignment_cardiac = """You will be given a single claim explaining why a patient was predicted to be at high or low risk of experiencing cardiac arrest within the next {} (Yes/No). You will also be given a series of categories that an expert clinician would use to perform determine if there is high risk of imminent cardiac arrest.

Your task is as follows:
1. Determine which expert category is most aligned with the claim. 
2. Rate how strongly the category aligns with the claim on a scale of 0-1 (0 being lowest, 1 being highest. Use increments of 0.1). 

Return your answer as:
Category: <category>
Category Alignment Rating: <rating>
Reasoning: <A brief explanation of why you selected the chosen category and why you judged the alignment rating as you did.>

-----
Expert categories:
1. Extreme Tachyarrhythmias: Sustained ventricular tachycardia or very rapid ventricular rates (especially in a patient with left ventricular dysfunction) can precipitate hemodynamic collapse and frequently degenerate into ventricular fibrillation, resulting in sudden cardiac death if uncorrected.
2. Ventricular Ectopy/NSVT: Runs of non-sustained ventricular tachycardia or frequent premature ventricular contractions in a critically ill patient signal electrical instability and portend a high risk of progression to a full cardiac arrest, particularly in the setting of underlying coronary disease or cardiomyopathy
3. Bradycardia or Heart-Rate Drop: The onset of significant bradycardia or a sudden ≥30% decline in heart rate is a well-documented precursor to in-hospital cardiac arrest (often preceding pulseless electrical activity or asystole) and should be treated as an alarm sign
4. Depressed Heart Rate Variability: An abnormally low short-term heart rate variability (minimal fluctuation in R–R intervals) reflects autonomic dysfunction and has been linked to impending cardiac arrest in ICU patients, serving as a strong predictor of near-term code events
5. QRS Widening (Conduction Delay): New or progressive prolongation of the QRS duration on the ECG is an ominous finding in the ICU, often observed in the minutes before cardiac arrest and associated with higher mortality due to deteriorating ventricular conduction
6. Dynamic ST-Segment Changes: Acute ischemic changes on continuous ECG (notably ST-segment elevation or depression) indicate ongoing myocardial infarction or injury and are treated as red flags for imminent ventricular fibrillation and cardiac arrest in the ICU setting
7. Prolonged QT Interval: A markedly prolonged QTc interval (whether drug-induced or due to congenital factors) creates a substrate for polymorphic ventricular tachycardia (torsades de pointes), which can quickly degenerate into ventricular fibrillation and cause sudden cardiac arrest if not promptly addressed
8. Severe Hyperkalemia Signs: Electrocardiographic signs of severe hyperkalemia (such as peaked T-waves, loss of P-waves, and a widening QRS complex) herald an impending arrest – as potassium levels rise, the ECG may evolve to a sine-wave pattern and typically culminate in ventricular fibrillation or asystole without immediate intervention
9. Electrical Alternans: The presence of electrical alternans (beat-to-beat alternating QRS amplitude) on ECG is highly specific for cardiac tamponade, a critical condition that can precipitate abrupt pulseless arrest unless the pericardial effusion is emergently relieved.
10. Advanced Age: Increasing age is a major risk factor for cardiac arrest (events are very rare in patients under 30), with older ICU patients being significantly more prone to sudden arrest
11. Male Sex: Male gender is associated with a higher incidence of cardiac arrest, as most cardiac arrests occur in men (with women’s risk rising post-menopause).
12. Underlying Cardiac Disease: The presence of serious cardiac conditions – such as coronary artery disease (especially a recent myocardial infarction) or severe heart failure – greatly elevates short-term cardiac arrest risk by creating an electrically and hemodynamically unstable myocardium.
13. Critical Illness (Sepsis/Shock): Severe sepsis or septic shock substantially raises the likelihood of cardiac arrest in the near term by causing hypoxia, hypotension, and metabolic derangements that often lead to pulseless electrical activity or asystole.
-----

Here are some examples:
[Example 1]
Claim:  A skin lesion of the scalp is a condition not directly related to cardiac function.
Category: Critical Illness (Sepsis/Shock)
Category Alignment Rating: 0.2
Reasoning: While a scalp lesion is not directly cardiac-related, if interpreted as a possible sign of infection or systemic compromise (e.g., an infected wound in a septic patient), it could weakly align with the critical illness category. However, without explicit signs of sepsis or shock, the connection remains speculative, hence the low alignment rating.

[Example 2]
Claim: The irregularity in the ECG could indicate a dangerous arrhythmia, such as ventricular tachycardia or fibrillation.
Category: Extreme Tachyarrhythmias
Category Alignment Rating: 0.9
Reasoning: The claim directly references dangerous arrhythmias such as ventricular tachycardia and fibrillation, which are hallmark indicators of the Extreme Tachyarrhythmias category. These arrhythmias are known precursors to sudden cardiac arrest. While the claim does not specify the duration or ventricular dysfunction context, the alignment is still very strong due to the mention of the precise arrhythmias characteristic of this category.

[Example 3]
Claim: The ECG irregularities are suggested by inconsistent waveform intervals and amplitudes.
Category: Depressed Heart Rate Variability
Category Alignment Rating: 0.7
Reasoning: The claim emphasizes "inconsistent waveform intervals and amplitudes," which most closely aligns with irregular R–R intervals, a hallmark of heart rate variability analysis. Although the claim lacks specificity (e.g., it doesn’t explicitly mention low variability or autonomic dysfunction), it implies irregular timing, which is a key aspect of depressed HRV as a precursor to cardiac arrest. Thus, the match is moderate to strong but not perfect due to the vagueness of the description.

[Example 4]
Claim: The patient is 86 years old. 
Category: Advanced Age
Category Alignment Rating: 1.0
Reasoning: The claim directly references advanced age, which this category identifies as a major risk factor for cardiac arrest.

[Example 5]
Claim: The admission wasn't due to cardiac issues.
Category: Underlying Cardiac Disease
Category Alignment Rating: 0.5
Reasoning: The claim highlights the absence of underlying cardiac disease, which is the inverse of a known risk factor for cardiac arrest. While the expert category focuses on increased risk due to the presence of cardiac disease, this claim indirectly relates to it by implying a potentially lower risk. The alignment is moderate because the claim addresses the category by exclusion rather than direct evidence of risk.

Now, determine the category and alignment rating for the following claim:
Claim: {}
"""
