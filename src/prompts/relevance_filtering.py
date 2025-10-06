import PIL.Image
from pathlib import Path

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
3. Is not a general statement about surgical practices, but rather a specific claim about a feature in the image.

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation pointing to the visual feature and criterion that supports your judgment>

I will now give a few examples so you get the hang of it.
"""

def load_relevance_cholec_prompt(image, claim: str):
    image1 = PIL.Image.open(Path(__file__).parent / "data" / "cholec_fewshot_1_image.png")
    image1.load()
    image1_claim = "The safe region is located in the upper central portion of the surgical field, where there is clear visualization of the fatty tissue surrounding the gallbladder."
    image1_reasoning = "This gives an accurate and specific description of the safe region in the image."
    image1_relevance = "Yes"

    image2 = PIL.Image.open(Path(__file__).parent / "data" / "cholec_fewshot_2_image.png")
    image2.load()
    image2_claim = "The tissue here appears pale pink to yellowish, indicating the gallbladder's serosa and underlying wall."
    image2_reasoning = "This is an accurate and specific description of a visual feature in the image."
    image2_relevance = "Yes"

    image5 = PIL.Image.open(Path(__file__).parent / "data" / "cholec_fewshot_5_image.png")
    image5.load()
    image5_claim = "The surgeon should maintain dissection within the safe zone, working methodically to establish the critical view of safety before any structures are divided."
    image5_reasoning = "This is generic advice that is not specific to the image."
    image5_relevance = "No"

    image6 = PIL.Image.open(Path(__file__).parent / "data" / "cholec_fewshot_6_image.png")
    image6.load()
    image6_claim = "The safe zone to dissect is above the Rouviere's sulcus, a known landmark for safe dissection."
    image6_reasoning = "There is no Rouviere's sulcus visible in this image."
    image6_relevance = "No"

    image10 = PIL.Image.open(Path(__file__).parent / "data" / "cholec_fewshot_10_image.png")
    image10.load()
    image10_claim = "This area demonstrates appropriate tissue separation and appears to be free of major vascular or biliary structures."
    image10_reasoning = "This image shows the starting phase of the surgery where the tissue is not separated yet."
    image10_relevance = "No"

    return (relevance_cholec,
        "[Example 1]",
        "Input:", image1,
        "Claim:", image1_claim,
        "Relevance:", image1_relevance,
        "Reasoning:", image1_reasoning,

        "[Example 2]",
        "Input:", image2,
        "Claim:", image2_claim,
        "Relevance:", image2_relevance,
        "Reasoning:", image2_reasoning,

        "[Example 3]",
        "Input:", image5,
        "Claim:", image5_claim,
        "Relevance:", image5_relevance,
        "Reasoning:", image5_reasoning,

        "[Example 4]",
        "Input:", image6,
        "Claim:", image6_claim,
        "Relevance:", image6_relevance,
        "Reasoning:", image6_reasoning,

        "[Example 5]",
        "Input:", image10,
        "Claim:", image10_claim,
        "Relevance:", image10_relevance,
        "Reasoning:", image10_reasoning,

        "Now evaluate the following",
        "Input: ", image,
        "Claim: ", claim,
    )

relevance_massmaps = """
You will be given:
- An image of a weak lensing mass map (the "Input").
- Its prediction for Omega_m and sigma_8 (the "Output").
- A textual Claim describing visual information in that image potentially related to its prediction of Omega_m and sigma_8 (the "Claim").

A claim is relevant if and only if:
1. It refers to a visually detectable feature in the image.
2. It pertains to predicting Omega_m and sigma_8 based on the visual information in the image.
3. Is not a general statement about weak lensing mass maps, but rather a specific claim about a feature in the image.
4. When unsure, lean towards "Yes" if the claim is not wrong to the image to be more inclusive.

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation pointing to the visual feature and criterion that supports your judgment>

I will now give a few examples so you get the hang of it.
"""

def load_relevance_massmaps_prompt(image, output, claim: str):
    image1 = PIL.Image.open(Path(__file__).parent / "data" / "massmaps_relevance_few_shot_examples.0.Omega0.1041.sigma0.9396.png")
    image1.load()
    image1_output = "Omega_m = 0.1041, sigma_8 = 0.9396"
    image1_claim = "The balance between blue and gray regions and red and yellow regions in the map indicates a moderate matter density and fluctuation levels."
    image1_reasoning = "The image contains a large amount of blue and gray regions, which are underdense areas. Therefore, the underdense and overdense regions are not balanced. The claim is not supported by but contradicts the image."
    image1_relevance = "No"

    image2 = PIL.Image.open(Path(__file__).parent / "data" / "massmaps_relevance_few_shot_examples.1.Omega0.2.sigma1.15.png")
    image2.load()
    image2_output = "Omega_m = 0.2, sigma_8 = 1.15"
    image2_claim = "There exist a large amount of yellow regions in the map, indicating clustering."
    image2_reasoning = "The image does contain a significant amount of yellow regions, which are overdense areas. This is relevant information for predicting sigma_8, and the claim is supported by the image."
    image2_relevance = "Yes"

    image3 = PIL.Image.open(Path(__file__).parent / "data" / "massmaps_relevance_few_shot_examples.2.Omega0.3586.sigma0.9762.png")
    image3.load()
    image3_output = "Omega_m = 0.3586, sigma_8 = 0.9762"
    image3_claim = "Voids are generally large low density regions in space."
    image3_reasoning = "This is background knowledge, not derived from the data."
    image3_relevance = "No"

    image4 = PIL.Image.open(Path(__file__).parent / "data" / "massmaps_relevance_few_shot_examples.3.Omega0.4612.sigma0.5614.png")
    image4.load()
    image4_output = "Omega_m = 0.4612, sigma_8 = 0.5614"
    image4_claim = "The weak lensing map shows a mix of blue, gray, red, and some yellow regions."
    image4_reasoning = "This claim is not wrong to the image, but it only states very naive information about the colors distributed in the map, without saying what the specificdistribution is like. This is not useful to cosmologists because they are not interpretable features for them."
    image4_relevance = "No"

    image5 = image1
    image5.load()
    image5_output = image1_output
    image5_claim = "There are mostly blue and gray regions spread out in the map, especially in the lower left corner."
    image5_reasoning = "The image contains a large amount of blue and gray regions, which are underdense areas, and indeed in the lower left corner. Therefore, the underdense and overdense regions are not balanced. The claim is supported by the image."
    image5_relevance = "Yes"

    image6 = image2
    image6.load()
    image6_output = image2_output
    image6_claim = "The presence of highly concentrated peaks or clusters in the map indicates a relatively high sigma_8."
    image6_reasoning = "This claim is talking about peaks and clusters which are interpretable information to cosmologists. Also, the peaks/clusters actually present in the image."
    image6_relevance = "Yes"
    
    return (relevance_massmaps,
        "[Example 1]",
        "Input:", image1,
        "Output:", image1_output,
        "Claim:", image1_claim,
        "Relevance:", image1_relevance,
        "Reasoning:", image1_reasoning,

        "[Example 2]",
        "Input:", image2,
        "Output:", image2_output,
        "Claim:", image2_claim,
        "Relevance:", image2_relevance,
        "Reasoning:", image2_reasoning,

        "[Example 3]",
        "Input:", image3,
        "Output:", image3_output,
        "Claim:", image3_claim,
        "Relevance:", image3_relevance,
        "Reasoning:", image3_reasoning,

        "[Example 4]",
        "Input:", image4,
        "Output:", image4_output,
        "Claim:", image4_claim,
        "Relevance:", image4_relevance,
        "Reasoning:", image4_reasoning,

        "[Example 5]",
        "Input:", image5,
        "Output:", image5_output,
        "Claim:", image5_claim,
        "Relevance:", image5_relevance,
        "Reasoning:", image5_reasoning,

        "[Example 6]",
        "Input:", image6,
        "Output:", image6_output,
        "Claim:", image6_claim,
        "Relevance:", image6_relevance,
        "Reasoning:", image6_reasoning,

        "Now evaluate the following",
        "Input: ", image,
        "Output: ", output,
        "Claim: ", claim,
    )

# relevance_massmaps = """You will be given an image of a weak lensing mass map, its prediction for Omega_m and sigma_8, and a claim that may or may not be relevant to an explanation of the prediction. Your task is to decide whether the claim is relevant to explaining the prediction for this specific mass map.

# A claim is relevant if and only if:
# (1) It is supported by the content of the mass map (i.e., it does not hallucinate or speculate beyond what is said).
# (2) It helps explain why the mass map received the given prediction (i.e., it directly relates to the mass map's features, such as the distribution of mass, the presence of voids or clusters, or the overall structure of the map).

# Return your answer as:
# Relevance: <Yes/No>
# Reasoning: <A brief explanation of your judgment, pointing to specific support or lack thereof>

# Here are some examples:

# [Example 1]
# Input: (Image 1)
# Output: Omega_m = 0.1041, sigma_8 = 0.9396
# Claim: The balance between blue and gray regions and red and yellow regions in the map indicates a moderate matter density and fluctuation levels.
# Relevance: No
# Reasoning: The image contains a large amount of blue and gray regions, which are underdense areas. Therefore, the underdense and overdense regions are not balanced. The claim is not supported by but contradicts the image.

# [Example 2]
# Input: (Image 2)
# Output: Omega_m = 0.2, sigma_8 = 1.15
# Claim: There exist a large amount of yellow regions in the map, which indicates a relatively high sigma_8.
# Relevance: Yes
# Reasoning: The image does contain a significant amount of yellow regions, which are overdense areas. This is relevant information for predicting the sigma_8, and the claim is supported by the image.

# [Example 3]
# Input: (Image 3)
# Output: Omega_m = 0.3586, sigma_8 = 0.9762
# Claim: Voids are large low density regions in space.
# Relevance: No
# Reasoning: This is background knowledge, not derived from the data.

# [Example 4]
# Input: (Image 4)
# Output: Omega_m = 0.4612, sigma_8 = 0.5614
# Claim: The weak lensing map shows a mix of blue, gray, red, and some yellow regions.
# Relevance: No
# Reasoning: This claim is not wrong to the image, but it only states very naive information about the colors distributed in the map, without saying what are these blue, gray, red, and yellow regions in the eyes of cosmologists, and how the distribution is like. This is not useful to cosmologists because they are not interpretable features for them.

# [Example 5]
# Input: (Image 2)
# Output: Omega_m = 0.2, sigma_8 = 1.15
# Claim: In the lower right corner, there is a large concentration of yellow region which are clusters.
# Relevance: Yes
# Reasoning: This claim is correct according to the image as there are indeed a large concentration of yellow regions in the lower right corner of the image, which are clusters.

# [Example 6]
# Input: (Image 2)
# Output: Omega_m = 0.2, sigma_8 = 1.15
# Claim: In the lower right corner, there is a large concentration of blue and gray regions which are voids.
# Relevance: No
# Reasoning: This claim is wrong according to the image as the lower right corner of the image is actually a large concentration of yellow regions which are clusters, not blue and gray regions which are voids.

# [Example 7]
# Input: (Image 1)
# Output: Omega_m = 0.1041, sigma_8 = 0.9396
# Claim: There are mostly blue and gray regions spread out in the map, especially in the lower left corner.
# Relevance: Yes
# Reasoning: The image contains a large amount of blue and gray regions, which are underdense areas, and indeed in the lower left corner. Therefore, the underdense and overdense regions are not balanced. The claim is supported by the image.

# [Example 8]
# Input: (Image 1)
# Output: Omega_m = 0.1041, sigma_8 = 0.9396
# Claim: There is a large amount of clusters in the middle of the map.
# Relevance: Yes
# Reasoning: The image contains a large amount of yellow regions, which are clusters, and indeed in the middle of the map. Therefore, the underdense and overdense regions are not balanced. The claim is supported by the image.

# Now, determine whether the following claim is relevant to the given mass map and prediction:
# Input: (Image 5)
# Output: {}
# Claim: {}
# """

relevance_emotion = """You will be given a Reddit comment, its emotion label, and a claim that may or may not be relevant to an explanation of the emotion label. Your task is to decide whether the claim is relevant to explaining the emotion label for the specific text comment.

A claim is relevant if and only if:
(1) It is supported by the content of the utterance (i.e., it does not hallucinate or speculate beyond what is said).
(2) It helps explain why the utterance received the given emotion label (i.e., it directly relates to tone, phrasing, sentiment, or other aspects relevant to the label).

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation of your judgment, pointing to specific support or lack thereof>

Here are some examples:

[Example 1]
Text: Apologies, I take it all back as I’ve just seen his latest effort
Emotion Label: remorse
Claim: The speaker shows regret and self‑reproach
Relevance: Yes
Reasoning: The speaker is explicitly apologizing and retracting their previous statement, which directly indicates remorse. The claim about the speaker showing regret and self-reproach aligns with the tone and content of the utterance.

[Example 2]
Text: At least it’s not anything worse, and that you are still close to that person :)
Label: relief
Claim: The speaker is expressing happiness at a positive outcome.
Relevance: No
Reasoning: This claim is not relevant as it does not relate to relief. The claim speculates a different emotion (happiness) that doesn't directly relate to the sentiment or tone of the given text.

[Example 3]
Text: seriously wtf. I want to see how the whole hand went in detail. that was the sickest soulread ever
Label: admiration
Claim: The exclamations (“seriously wtf… sickest soulread ever”) show astonished praise for an impressive play.
Relevance: Yes
Reasoning: The exclamations "seriously wtf" and "sickest soulread ever" express strong admiration and astonishment, which directly supports the emotion label of admiration. The claim accurately describes the tone of praise and amazement present in the utterance.

Now, determine whether the following claim is relevant to the given text and emotion label:
Text: {}
Emotion Label: {}
Claim: {}
"""

relevance_supernova = """You will be given a dataset of flux measurement values over time across multiple wavelengths, and a claim that may or may not be relevant to explaining what astrophysical class the dataset represents. Your task is to decide whether the claim is relevant to explaining the astrophysical classification, where the possible classification categories include: RR-Lyrae (RRL), peculiar type Ia supernova (SNIa-91bg), type Ia supernova (SNIa), superluminous supernova (SLSN-I), type II supernova (SNII), microlens-single (mu-Lens-Single), eclipsing binary (EB), M-dwarf, kilonova (KN), tidal disruption event (TDE), peculiar type Ia supernova (SNIax), type Ibc supernova (SNIbc), Mira variable, and active galactic nuclei (AGN).
A claim is relevant if and only if:
(1) It is directly supported by the time-series data (i.e., it refers to trends, changes, or patterns in flux across time and wavelengths).
(2) It helps explain why the model predicted this specific astrophysical class (i.e., it highlights characteristics that distinguish this class from others).

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation of your judgment, pointing to specific support or lack thereof>

I will now give a few examples so you get the hang of it.
"""

def load_relevance_supernova_prompt(image, output, claim: str):
    image1 = PIL.Image.open(Path(__file__).parent / "data" / "supernova_11_0.png")
    image1.load()
    image1_output = "type Ia supernova (SNIa)"
    image1_claim = "The time series shows a rapid increase in brightness followed by a gradual decline."
    image1_reasoning = "This is a specific flux pattern observable in the data, characteristic of certain supernova types like SNIa."
    image1_relevance = "Yes"

    image2 = PIL.Image.open(Path(__file__).parent / "data" / "supernova_3_0.png")
    image2.load()
    image2_output = "type II supernova (SNII)"
    image2_claim = "The dataset represents a time series of observations for a astrophysical event."
    image2_reasoning = "This is a general statement and does not support any specific classification."
    image2_relevance = "No"

    image3 = PIL.Image.open(Path(__file__).parent / "data" / "supernova_2_0.png")
    image3.load()
    image3_output = "eclipsing binary (EB)"
    image3_claim = "Specific wavelengths such as 7545.98 Å, 8590.90 Å, and 9710.28 Å are present in the data."
    image3_reasoning = "Although multiple wavelengths are present, simply listing them does not contribute to classification."
    image3_relevance = "No"

    image4 = PIL.Image.open(Path(__file__).parent / "data" / "supernova_8_0.png")
    image4.load()
    image4_output = "M-dwarf"
    image4_claim = "All the measurement are around 0."
    image4_reasoning = "This contradicts with the given time series data since it has a peak around time 60250."
    image4_relevance = "No"

    image5 = image1
    image5.load()
    image5_output = image1_output
    image5_claim = "Type Ia supernovae are valuable as standard candles for measuring cosmic distances."
    image5_reasoning = "This is background knowledge and not derived from the observed data."
    image5_relevance = "No"

    image6 = image2
    image6.load()
    image6_output = image2_output
    image6_claim = "The time series shows a rapid decrease in measurement value for wavelength 3670."
    image6_reasoning = "This claim is talking about trend in value which are interpretable information to astrophysicists. Also, the trend actually present in the time series data."
    image6_relevance = "Yes"
    
    return (relevance_supernova,
        "[Example 1]",
        "Input:", image1,
        "Output:", image1_output,
        "Claim:", image1_claim,
        "Relevance:", image1_relevance,
        "Reasoning:", image1_reasoning,

        "[Example 2]",
        "Input:", image2,
        "Output:", image2_output,
        "Claim:", image2_claim,
        "Relevance:", image2_relevance,
        "Reasoning:", image2_reasoning,

        "[Example 3]",
        "Input:", image3,
        "Output:", image3_output,
        "Claim:", image3_claim,
        "Relevance:", image3_relevance,
        "Reasoning:", image3_reasoning,

        "[Example 4]",
        "Input:", image4,
        "Output:", image4_output,
        "Claim:", image4_claim,
        "Relevance:", image4_relevance,
        "Reasoning:", image4_reasoning,

        "[Example 5]",
        "Input:", image5,
        "Output:", image5_output,
        "Claim:", image5_claim,
        "Relevance:", image5_relevance,
        "Reasoning:", image5_reasoning,

        "[Example 6]",
        "Input:", image6,
        "Output:", image6_output,
        "Claim:", image6_claim,
        "Relevance:", image6_relevance,
        "Reasoning:", image6_reasoning,

        "Now evaluate the following",
        "Input: ", image,
        "Output: ", output,
        "Claim: ", claim,
    )
    
relevance_sepsis = """You will be given a time-series EHR(Electronic Health Record) data from the first 2 hours of the ED(Emergency Department) admission that includes the name of a measurement or medication and its corresponding value over time. You will also be given a binary prediction of whether a patient is at high risk of developing sepsis within the next 12 hours, and a claim that may or may not be relevant to explaining why the sepsis prediction was assigned. Your task is to decide whether the claim is relevant to explaining the patient’s sepsis prediction for the given time series data.

A claim is relevant if and only if:
(1) It is directly supported by the time-series data (i.e.,reference to a specific value, trend, or change in a measurement such as heart rate or temperature over time).
(2) It helps explain why the model predicted this specific class (i.e., it contributes to explaining why the model predicted the specified class (yes/no), based on known sepsis indicators (i.e., organ dysfunction, suspected infection, SOFA criteria, vital sign abnormalities)).

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation of your judgment, pointing to specific support or lack thereof>

Here are some examples:

[Example 1]
Data: '36.0: ONDANSETRON HCL (PF) 4 MG/2 ML INJ SOLN_ 53.0, WAM DIFTYP: AUTO; 53.0: IMMATURE GRANULOCYTE_ ABSOLUTE (AUTO DIFF) WAM, .05 K/uL; 53.0: LYMPHOCYTE_ ABSOLUTE (AUTO DIFF), 1.23 K/uL; 53.0: EOSINOPHIL_ ABSOLUTE (AUTO DIFF), .24 K/uL; 53.0: LYMPHOCYTE % (AUTO DIFF), 9.9 %; 53.0: RED CELL DISTRIBUTION WIDTH (RDW), 14.0 %; 53.0: NRBC_ PERCENT (HEMATOLOGY), .0 %; 53.0: MEAN CORPUSCULAR HEMOGLOBIN (MCH), 27.8 pg; 53.0: MEAN CORPUSCULAR VOLUME (MCV), 85.1 fL; 53.0: MEAN CORPUSCULAR HEMOGLOBIN CONCENTRATION (MCHC), 32.7 g/dL; 53.0: MONOCYTE_ ABSOLUTE (AUTO DIFF), .82 K/uL; 53.0: MONOCYTE % (AUTO DIFF), 6.6 %; 53.0: PLATELET COUNT (PLT), 230 K/uL; 53.0: HEMOGLOBIN (HGB), 11.6 g/dL; 53.0: BASOPHIL % (AUTO DIFF), .3 %; 53.0: NRBC_ ABSOLUTE (HEMATOLOGY), .00 K/uL; 53.0: NEUTROPHIL_ ABSOLUTE (AUTO DIFF), 10.07 K/uL; 53.0: BASOPHIL_ ABSOLUTE (AUTO DIFF), .04 K/uL; 53.0: EOSINOPHIL % (AUTO DIFF), 1.9 %; 53.0: NEUTROPHIL % (AUTO DIFF), 80.9 %; 53.0: WHITE BLOOD CELLS (WBC), 12.5 K/uL; 53.0: IMMATURE GRANULOCYTE % (AUTODIFF) WAM, .4 %; 53.0: RED BLOOD CELLS (RBC), 4.17 MIL/uL; 53.0: HEMATOCRIT (HCT), 35.5 %; 57.0: HYDROMORPHONE 1 MG/ML INJ SYRG_ 57.0, LR IV BOLUS - 500 ML; 81.0: CO2, 22 mmol/L; 81.0: GLUCOSE, 95 mg/dL; 81.0: SODIUM, 139 mmol/L; 81.0: ANION GAP, 12 mmol/L; 81.0: BILIRUBIN_ TOTAL, .3 mg/dL; 81.0: GLOBULIN, 2.8 g/dL; 81.0: PROTEIN_ TOTAL, 6.5 g/dL; 81.0: MAGNESIUM, 2.0 mg/dL; 81.0: ALKALINE PHOSPHATASE, 92 U/L; 81.0: ALT (SGPT), 25 U/L; 81.0: POTASSIUM, 4.0 mmol/L; 81.0: AST (SGOT), 27 U/L; 81.0: ALBUMIN, 3.7 g/dL; 81.0: LIPASE, 19 U/L; 81.0: CHLORIDE, 105 mmol/L; 81.0: CALCIUM, 9.1 mg/dL; 81.0: CORRECTED CALCIUM, 9.3 mg/dL; 81.0: EGFR FOR AFRICAN AMERICAN, 95 mL/min/1.73 m2; 81.0: EGFR REFIT WITHOUT RACE (2021), 83 mL/min/1.73 m2; 81.0: CREATININE, .93 mg/dL; 81.0: BLOOD UREA NITROGEN (BUN), 16 mg/dL; 81.0: URIC ACID, 3.8 mg/dL; 108.0: GLUCOSE, URINE (UA): Negative; 108.0: PH, URINE (UA): 6.0; 108.0: NITRITE, URINE (UA): Negative; 108.0: SPECIFIC GRAVITY, URINE (UA): 1.016; 108.0: KETONE, URINE (UA): Negative; 0.0: Age, 32; 0.0: Gender, F; 0.0: Race, White; 0.0: Means_of_arrival, Self; 0.0: Triage_Temp, 36.7; 0.0: Triage_HR, 88.0; 0.0: Triage_RR, 18.0; 0.0: Triage_SBP, 136.0; 0.0: Triage_DBP, 100.0; 0.0: Triage_acuity, 3-Urgent; 0.0: CC, FLANK PAIN,NAUSEA'
Prediction: No
Claim: The dataset represents a time series of a person in ED.
Relevance: No
Reasoning: This is a general statement and does not justify any specific classification.

[Example 2]
Data: '64.0: RSV, Not Detected; 64.0: INFLUENZA B, Not Detected; 64.0: SARS-COV-2 RNA, Not Detected; 64.0: INFLUENZA A, Not Detected; 100.0: IBUPROFEN 800 MG PO TABS_ 100.0, ACETAMINOPHEN 500 MG PO TABS; 0.0: Age, 62; 0.0: Gender, F; 0.0: Race, Other; 0.0: Means_of_arrival, Self; 0.0: Triage_Temp, 39.5; 0.0: Triage_HR, 135.0; 0.0: Triage_RR, 18.0; 0.0: Triage_SBP, 111.0; 0.0: Triage_DBP, 64.0; 0.0: Triage_acuity, 2-Emergent; 0.0: CC, NECK PAIN,SORE THROAT,HEADACHE,JOINT PAIN'
Prediction: Yes
Claim: The patient is 62 years old, which is a moderate risk factor for sepsis.
Relevance: Yes
Reasoning: The age is directly supported by the data and is a known clinical risk factor for sepsis.

[Example 3]
Data: '0.0: Gender, Male; 0.0: AGE, 40-60; 0.0: GLUCOSE POINT OF CARE, 336.0; 0.02: WEIGHT/SCALE, 5440.0; 0.02: R IP IDEAL BODY WEIGHT, 68.4; 0.02: PULSE OXIMETRY, 92.0; 0.02: HEIGHT, 68.0; 0.02: PULSE, 84.0; 0.02: R AN ADJUSTED BODY WEIGHT, 102.73; 0.02: TEMPERATURE, 99.6; 0.02: RESPIRATIONS, 22.0; 0.05: R OR GLASGOW COMA SCALE SCORE, 15.0; 0.05: R OR GLASGOW COMA SCALE BEST MOTOR RESPONSE, 6.0; 0.05: R OR GLASGOW COMA SCALE EYE OPENING, 4.0; 0.05: R OR GLASGOW COMA SCALE BEST VERBAL RESPONSE, 5.0; 0.27: HEMOGLOBIN, 10.4; 0.27: MEAN CELLULAR HEMOGLOBIN CONCENTRATION, 32.0; 0.27: MEAN CELLULAR HEMOGLOBIN, 27.0; 0.27: # NEUTROPHILS, 16.1; 0.27: WBC, 18.7; 0.27: % NEUTROPHILS, 86.3; 0.27: PLATELETS, 276.0'
Prediction: Yes
Claim: The patient's condition warrants close monitoring and further investigation for infection.
Relevance: Yes
Reasoning: This claim includes signs of possible infection, and infection aligns with the model's rationale for predicting sepsis.

[Example 4]
Data: '33.0: Pain, 10.0; 38.0: HR, 116.0; 38.0: RR, 22.0; 38.0: SpO2, 93.0; 38.0: SBP, 92.0; 38.0: DBP, 61.0; 38.0: MAP, 71.3333333333333; 38.0: LPM_O2, 55.0; 38.0: Temp, 97.5; 93.0: INFLUENZA B, Not Detected; 93.0: RSV, Not Detected; 93.0: INFLUENZA A, Not Detected; 93.0: SARS-COV-2 RNA, Not Detected; 118.0: HR, 114.666666666667; 118.0: RR, 29.5; 118.0: 1min_HRV, 43.8493126767367; 118.0: SBP, 78.0; 118.0: DBP, 57.0; 118.0: MAP, 64.0; 119.0: HR, 112.965517241379; 119.0: RR, 24.3620689655172; 119.0: 1min_HRV, 78.3210127790084; 119.0: 5min_HRV, 75.2505983451634; 0.0: Age, 82; 0.0: Gender, M; 0.0: Race, White; 0.0: Means_of_arrival, Self; 0.0: Triage_Temp, 36.4; 0.0: Triage_HR, 116.0; 0.0: Triage_RR, 22.0; 0.0: Triage_SBP, 92.0; 0.0: Triage_DBP, 61.0; 0.0: Triage_acuity, 2-Emergent; 0.0: CC, ABDOMINAL PAIN'
Prediction: Yes
Claim: The patient exhibits several risk factors and early warning signs for sepsis.
Relevance: No
Reasoning: This is too vague. It does not specify what the risk factors or warning signs are, nor does it directly reference any values or patterns from the data. 

Now, determine whether the following claim is relevant to the given the time series data and the prediction label:
Input: {}
Output: {}
Claim: {}
"""


relevance_cardiac = """
You will be given:
- Basic background information about an ICU patient (age, gender, race, and primary reason for initial ICU admittance), and time-series Electrocardiogram (ECG) data plotted in a graph from the first {} of an ECG monitoring period during the patient's ICU stay, where the samples are taken at {} Hz (the "Input").
- A binary prediction of whether the patient is at high risk of experiencing cardiac arrest within the next {} (the "Prediction").
- A textual claim that may or may not be relevant to explaining why the cardiac arrest prediction was assigned (the "Claim"). 

Your task is to decide whether the claim is relevant to explaining the patient’s cardiac arrest prediction for the given patient background information and time series data.

A claim is relevant if and only if:
1. It is directly supported by the patient background information and time-series ECG data.
2. It helps explain why the model predicted yes/no.

Return your answer as:
Relevance: <Yes/No>
Reasoning: <A brief explanation of your judgment, pointing to specific support or lack thereof>

Here are a few examples so you can get the hang of it.
"""


def load_relevance_cardiac_prompt(background, image, llm_label, claim: str):
    example1_background = "The patient is age 79, gender F, race White, and was admitted to the ICU for Closed fracture of right femur, unspecified fracture morphology, unspecified portion of femur, initial encounter (CMS-HCC)."
    image1 = PIL.Image.open(Path(__file__).parent / "data" / "cardiac_relevance_fewshot_1_image_99593648_1.png")
    image1.load()
    example1_prediction = "Yes"
    example1_claim = "The ECG data shows noticeable irregularity with a prominent spike and abrupt shifts in the waveform around 60 seconds."
    example1_relevance = "Yes"
    example1_reasoning = "The claim highlights a sharp spike and abrupt waveform changes around 60 seconds, which are visible in the ECG and likely indicative of cardiac instability relevant to the high-risk prediction."

    example2_background = "The patient is age 39, gender F, race Other, and was admitted to the ICU for Strep pharyngitis."
    image2 = PIL.Image.open(Path(__file__).parent / "data" / "cardiac_relevance_fewshot_2_image_99877003_1.png")
    image2.load()
    example2_prediction = "No"
    example2_claim = "Ventricular fibrillation or sustained tachycardia are typical warning signs for an imminent cardiac arrest."
    example2_relevance = "No"
    example2_reasoning = "The claim describes general warning signs for cardiac arrest, but there is no evidence of ventricular fibrillation or sustained tachycardia in the provided ECG plot or background information, so it does not help explain the model's prediction for this specific patient."
    
    example3_background = "The patient is age 68, gender M, race Asian, and was admitted to the ICU for Acute hypoxemic respiratory failure (CMS-HCC)."
    image3 = PIL.Image.open(Path(__file__).parent / "data" / "cardiac_relevance_fewshot_3_image_99579278_1.png")
    image3.load()
    example3_prediction = "Yes"
    example3_claim = "Compromised respiratory function can potentially indicate compromised cardiac function."
    example3_relevance = "Yes"
    example3_reasoning = "The claim links compromised respiratory function—explicitly stated in the patient's ICU admission reason—to potential cardiac dysfunction, which is a medically supported relationship and helps contextualize the model’s positive prediction."

    example4_background = "The patient is age 47, gender M, race Other, and was admitted to the ICU for Alcoholic intoxication without complication (CMS-HCC)."
    image4 = PIL.Image.open(Path(__file__).parent / "data" / "cardiac_relevance_fewshot_4_image_99318009_1.png")
    image4.load()
    example4_prediction = "No"
    example4_claim = "The ECG data shows regular rhythm and amplitude patterns."
    example4_relevance = "Yes"
    example4_reasoning = "The claim is relevant because the ECG waveform in the later part of the 2-minute window appears stable with regular amplitude and rhythmic peaks, suggesting no acute arrhythmia or instability, which supports the model's prediction that the patient is not at high risk of cardiac arrest in the next 5 minutes."

    example5_background = "The patient is age 87, gender M, race White, and was admitted to the ICU for COVID-19."
    image5 = PIL.Image.open(Path(__file__).parent / "data" / "cardiac_relevance_fewshot_5_image_99972446_1.png")
    image5.load()
    example5_prediction = "Yes"
    example5_claim = "Being 87 years old places the patient in a higher risk category for cardiac events."
    example5_relevance = "Yes"
    example5_reasoning = "The patient’s advanced age of 87 is directly supported by the background information and is a medically recognized factor that advanced age contributes to increased vulnerability to cardiac arrest, making it relevant for explaining the model’s positive prediction."
    
    
    return (relevance_cardiac,
        "[Example 1]",
        "Input:", example1_background, image1,
        "Prediction:", example1_prediction,
        "Claim:", example1_claim,
        "Relevance:", example1_relevance,
        "Reasoning:", example1_reasoning,

        "[Example 2]",
        "Input:", example2_background, image2,
        "Prediction:", example2_prediction,
        "Claim:", example2_claim,
        "Relevance:", example2_relevance,
        "Reasoning:", example2_reasoning,
            
        "[Example 3]",
        "Input:", example3_background, image3,
        "Prediction:", example3_prediction,
        "Claim:", example3_claim,
        "Relevance:", example3_relevance,
        "Reasoning:", example3_reasoning,

        "[Example 4]",
        "Input:", example4_background, image4,
        "Prediction:", example4_prediction,
        "Claim:", example4_claim,
        "Relevance:", example4_relevance,
        "Reasoning:", example4_reasoning,

        "[Example 5]",
        "Input:", example5_background, image5,
        "Prediction:", example5_prediction,
        "Claim:", example5_claim,
        "Relevance:", example5_relevance,
        "Reasoning:", example5_reasoning,

        "Now evaluate the following",
        "Input:", background, image,
        "Prediction:", llm_label,
        "Claim: ", claim
    )