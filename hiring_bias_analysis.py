# %% [markdown]
# # AI Hiring Bias: A Causal Decomposition Analysis
#
# This project investigates whether an AI hiring tool's scores reflect
# genuine qualifications or encode bias against protected attributes — and
# if bias is present, how much of it flows through the AI score itself
# versus happening some other way.
#
# **Questions I'm answering, in the order I actually worked through them:**
# 1. What is the AI's bias score actually weighting — legitimate
#    qualification signals, or attributes that double as demographic
#    proxies (or even the protected attributes directly)?
# 2. Across protected attributes (gender, university tier, education,
#    age), which one has the largest gap in hiring outcomes that flows
#    *through* the AI score specifically, and does that finding hold up
#    under stress-testing?
# 3. At the individual level, how many candidates sitting right at the
#    hire/no-hire boundary would have gotten a different outcome if their
#    AI score had been an average score instead of their actual one?
#
# **A note on method:** I'm using two different tools that answer two
# different kinds of questions, and I want to be precise about which is
# which:
# - **SHAP** (Question 1) tells me what a model is *paying attention to* —
#   it's an explainability tool, not a causal one.
# - **DoWhy** (Question 2) tells me what would *actually change* if a
#   variable were different — it's built for causal effect estimation,
#   with a proper causal graph, an identification step, and refutation
#   tests to stress-test the result.
# Using both, and not confusing what each one can claim, is the point of
# this project.

# %% [markdown]
# ---
# ## Step 1: Import the modules

# %%
# Import the modules
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from dowhy import CausalModel
import shap
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Step 2: Read the data into a Pandas DataFrame

# %%
# Read the CSV file into a Pandas DataFrame
hiring_df = pd.read_csv('AI_Hiring_Bias_Dataset.csv')

# Review the DataFrame
hiring_df.head()

# %% [markdown]
# ## Step 3: Check the shape, data types, and missing values

# %%
# Check how many rows and columns are in the data
hiring_df.shape

# %%
# Check the data type of each column
hiring_df.dtypes

# %%
# Count missing values in each column
hiring_df.isnull().sum()

# %% [markdown]
# **Answer:** No missing values, so no cleanup needed before moving on.

# %% [markdown]
# ## Step 4: Check the overall hire rate and group sizes

# %%
# Check what percentage of candidates were hired
hiring_df['hired'].value_counts(normalize=True)

# %%
# Check group sizes for each protected attribute
print(hiring_df['gender'].value_counts())
print()
print(hiring_df['university_tier'].value_counts())
print()
print(hiring_df['education_level'].value_counts())

# %% [markdown]
# ## Step 5: Check the raw hire rate gap by protected attribute
#
# Before any modeling, I want to see whether there's even a hiring gap to
# explain. This is just a raw comparison — it doesn't control for
# qualifications yet.

# %%
# Hire rate by gender
hiring_df.groupby('gender')['hired'].mean().round(3)

# %%
# Hire rate by university tier
hiring_df.groupby('university_tier')['hired'].mean().round(3)

# %%
# Hire rate by education level
hiring_df.groupby('education_level')['hired'].mean().round(3)

# %%
# Hire rate by age group (split at the median age)
median_age = hiring_df['age'].median()
hiring_df['age_group'] = hiring_df['age'].apply(lambda x: 'older' if x > median_age else 'younger')
hiring_df.groupby('age_group')['hired'].mean().round(3)

# %% [markdown]
# ## Step 6: Check whether qualifications actually differ by gender
#
# If hire rates differ but qualifications are the same, that's the setup
# that makes this worth investigating further — a hiring gap that isn't
# explained by a difference in skill.

# %%
# Compare average qualification scores by gender
hiring_df.groupby('gender')[['technical_skill_score', 'coding_test_score', 'years_experience']].mean().round(2)

# %% [markdown]
# **Answer:** There's a real hire rate gap by gender (Male 35.3% vs. Female
# 30.9% vs. Non-Binary 26.6%) and an even bigger one by university tier
# (Tier 1: 44.1% vs. Tier 3: 26.3%). Education is close to flat across
# levels, and age shows a gap (older 38.3% vs. younger 29.1%). Importantly,
# average qualification scores barely differ by gender — so the gender gap
# in hiring isn't explained by a gap in actual skill. That's exactly the
# kind of unexplained gap worth digging into with the AI score.

# %% [markdown]
# ---
# ## Question 1: What is the AI's bias score actually weighting?
#
# My plan: build a model that predicts `ai_bias_score` from all the
# candidate's other information — qualifications, and also the protected
# attributes and known proxies (`university_tier`, `is_female`, etc.).
# Then use SHAP to see which inputs the model relies on most. If a
# protected attribute (or an obvious proxy for one) shows up as a heavy
# contributor, that's a sign the AI score itself may be encoding bias, not
# just reflecting qualifications.

# %% [markdown]
# ## Step 7: Create numeric versions of the protected attributes
#
# SHAP and regression models need numbers, not text categories, so I'm
# creating simple numeric/dummy versions of each protected attribute.

# %%
# university_tier as a number (1 = best tier, 3 = worst)
hiring_df['university_tier_num'] = hiring_df['university_tier'].map({'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3})

# gender as two dummy columns (Male is the reference group, so it doesn't need its own column)
hiring_df['is_female'] = (hiring_df['gender'] == 'Female').astype(int)
hiring_df['is_non_binary'] = (hiring_df['gender'] == 'Non-Binary').astype(int)

# education as a dummy column (Bachelor's vs. an advanced degree)
hiring_df['is_bachelors'] = (hiring_df['education_level'] == "Bachelor's").astype(int)

# Review the new columns
hiring_df[['gender', 'is_female', 'is_non_binary', 'university_tier', 'university_tier_num', 'education_level', 'is_bachelors']].head()

# %% [markdown]
# ## Step 8: List the qualification variables
#
# These are the legitimate, skill-based variables that should be driving
# the AI score if it's working as intended.

# %%
# These are the qualification-based variables (not protected attributes)
qualification_vars = [
    'technical_skill_score', 'communication_score', 'aptitude_test_score',
    'coding_test_score', 'years_experience', 'project_count',
    'github_activity_score', 'certifications_count', 'employment_gap_months'
]

# %% [markdown]
# ## Step 9: Fit a model predicting the AI bias score
#
# The features are the qualification variables plus the protected
# attribute columns from Step 7. The target is `ai_bias_score` itself —
# I'm trying to explain what drives the AI's own score.

# %%
# Set up the features and target
shap_features = qualification_vars + ['university_tier_num', 'is_female', 'is_non_binary', 'is_bachelors']
X_shap = hiring_df[shap_features]
y_shap = hiring_df['ai_bias_score']

# Fit a linear regression model
bias_score_model = LinearRegression()
bias_score_model.fit(X_shap, y_shap)

# %% [markdown]
# ## Step 10: Run SHAP on the model
#
# For each feature, SHAP tells me how much it typically pushes the AI's
# score up or down. I'm looking at the average size of that push (ignoring
# direction) to rank which features matter most.

# %%
# Set up the SHAP explainer for a linear model
explainer = shap.LinearExplainer(bias_score_model, X_shap)
shap_values = explainer(X_shap)

# Build a table of the average absolute SHAP value per feature
shap_importance = pd.DataFrame({
    'feature': shap_features,
    'mean_abs_shap_value': abs(shap_values.values).mean(axis=0)
})

# Sort so the most influential features are at the top
shap_importance = shap_importance.sort_values('mean_abs_shap_value', ascending=False)
shap_importance

# %% [markdown]
# **Answer:** `technical_skill_score` is the top feature, which is what I'd
# want to see from a legitimate scoring system. But `university_tier_num`
# is a close second — ahead of `coding_test_score`, `aptitude_test_score`,
# and `years_experience`, all of which barely register. That's concerning,
# since university tier can act as a proxy for socioeconomic background
# rather than a pure skill measure. Even more concerning: `is_female` ranks
# fourth, ahead of several qualification variables — meaning gender itself,
# not just a proxy for it, appears to directly influence the AI's score.
# This sets up Question 2: if gender directly affects the AI score, how
# much of gender's effect on actual hiring outcomes flows through that
# score?

# %% [markdown]
# ---
# ## Groundwork for Question 2: setting up the causal comparison
#
# Before I can compare protected attributes, I need to define the causal
# structure I'm assuming, then estimate two things for each attribute:
# 1. The **total effect** on `hired` (using DoWhy, with qualifications as
#    confounders I control for).
# 2. A decomposition into a **direct effect** (bypassing the AI score) and
#    an **indirect effect** (flowing through `ai_bias_score`).
#
# **My causal assumptions:** a protected attribute (like gender) can affect
# hiring two ways — directly (e.g. straightforward discrimination in the
# process) or indirectly, by affecting the AI score, which then affects the
# hiring decision. The qualification variables are confounders: they affect
# both the AI score and the hiring decision, and I need to control for them
# so I'm not mistaking "differences in skill" for "bias."
#
# **A limitation worth stating honestly:** I'm only controlling for
# qualification variables as confounders — I'm not controlling for the
# *other* protected attributes when I analyze one of them (e.g. I don't
# control for university tier when analyzing gender). That means some of
# what I attribute to one protected attribute could actually be
# overlapping with another. This is a real limitation of this analysis, not
# something I'm claiming to have solved.

# %% [markdown]
# ## Step 11: Set up the treatment variable for each protected attribute
#
# For each attribute, I'm defining "treatment = 1" as the group with the
# *lower* raw hire rate from Step 5, so the sign of every effect I
# calculate means the same thing: negative = being in that group hurts
# your odds of being hired.

# %%
# Gender: compare Female (treatment) vs. Male (reference)
# Dropping Non-Binary here so this is a clean two-group comparison
gender_subset = hiring_df[hiring_df['gender'].isin(['Female', 'Male'])].copy()
gender_subset['treatment'] = (gender_subset['gender'] == 'Female').astype(int)

# %%
# University tier: compare Tier 3 (treatment) vs. Tier 1 (reference)
# Dropping Tier 2 here for the same reason
tier_subset = hiring_df[hiring_df['university_tier'].isin(['Tier 3', 'Tier 1'])].copy()
tier_subset['treatment'] = (tier_subset['university_tier'] == 'Tier 3').astype(int)

# %%
# Education: compare Bachelor's (treatment) vs. Master's or PhD (reference)
# No rows need to be dropped here
education_subset = hiring_df.copy()
education_subset['treatment'] = education_subset['is_bachelors']

# %%
# Age: compare younger (treatment) vs. older (reference), split at the median
age_subset = hiring_df.copy()
age_subset['treatment'] = (age_subset['age_group'] == 'younger').astype(int)

# %% [markdown]
# ## Step 12: Loop through each attribute and estimate the total effect with DoWhy
#
# For each attribute, I build a DoWhy causal model, identify the effect,
# and estimate it using the qualification variables as confounders. I'm
# saving the results in a list so I can compare them all in Question 2.

# %%
# Package up the four attribute subsets so I can loop through them
attribute_subsets = {
    'gender (Female vs. Male)': gender_subset,
    'university_tier (Tier 3 vs. Tier 1)': tier_subset,
    'education (Bachelor\'s vs. advanced degree)': education_subset,
    'age (younger vs. older)': age_subset
}

# This will hold the results for each attribute
total_effect_results = []

for attribute_name, subset_df in attribute_subsets.items():

    causal_model = CausalModel(
        data=subset_df,
        treatment='treatment',
        outcome='hired',
        common_causes=qualification_vars
    )

    identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
    estimate = causal_model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

    total_effect_results.append({
        'attribute': attribute_name,
        'total_effect': estimate.value
    })

    print(f"{attribute_name}: total effect = {round(estimate.value, 4)}")

# %% [markdown]
# ## Step 13: Decompose each attribute's total effect into direct vs. indirect
#
# For each attribute, I fit two logistic regression models:
# - **Model A** predicts `hired` from treatment + confounders (the mediator
#   is left out) — the treatment's coefficient here is the total effect.
# - **Model B** predicts `hired` from treatment + confounders + the AI bias
#   score — the treatment's coefficient here is the direct effect, since
#   the AI score's contribution has been separated out.
#
# The indirect (mediated) effect is just the difference between the two.

# %%
# This will hold the decomposition results for each attribute
mediation_results = []

for attribute_name, subset_df in attribute_subsets.items():

    X_total = subset_df[['treatment'] + qualification_vars]
    X_direct = subset_df[['treatment', 'ai_bias_score'] + qualification_vars]
    y = subset_df['hired']

    model_total = LogisticRegression(max_iter=1000, random_state=1)
    model_total.fit(X_total, y)
    total_coef = model_total.coef_[0][0]

    model_direct = LogisticRegression(max_iter=1000, random_state=1)
    model_direct.fit(X_direct, y)
    direct_coef = model_direct.coef_[0][0]

    indirect_coef = total_coef - direct_coef
    proportion_mediated = indirect_coef / total_coef

    mediation_results.append({
        'attribute': attribute_name,
        'total_effect_coef': total_coef,
        'direct_effect_coef': direct_coef,
        'indirect_effect_coef': indirect_coef,
        'proportion_mediated': proportion_mediated
    })

mediation_table = pd.DataFrame(mediation_results)
mediation_table

# %% [markdown]
# ---
# ## Question 2: Which attribute has the largest algorithmically-mediated gap?

# %% [markdown]
# ## Step 14: Sort the table to find the largest mediated effect

# %%
# Sort by the size of the indirect (mediated) effect, largest first
mediation_table.sort_values('indirect_effect_coef', key=abs, ascending=False)

# %% [markdown]
# ## Step 15: Stress-test the top finding with a refutation test
#
# DoWhy has built-in ways to check whether an estimate is trustworthy. I'm
# running a placebo test: it swaps in a fake, randomly-generated treatment
# instead of the real one, and re-estimates the effect. If the estimate
# drops to roughly zero, that's a good sign — it means my original estimate
# wasn't just an artifact of how the model happens to fit random noise.

# %%
# Identify which attribute had the largest mediated effect, and re-run
# DoWhy on that one specifically for the refutation test
top_attribute = mediation_table.sort_values('indirect_effect_coef', key=abs, ascending=False).iloc[0]['attribute']
print(f"Attribute with the largest mediated effect: {top_attribute}")

top_subset = attribute_subsets[top_attribute]

top_causal_model = CausalModel(
    data=top_subset,
    treatment='treatment',
    outcome='hired',
    common_causes=qualification_vars
)
top_identified_estimand = top_causal_model.identify_effect(proceed_when_unidentifiable=True)
top_estimate = top_causal_model.estimate_effect(top_identified_estimand, method_name="backdoor.linear_regression")

refutation = top_causal_model.refute_estimate(
    top_identified_estimand,
    top_estimate,
    method_name="placebo_treatment_refuter"
)
print(refutation)

# %% [markdown]
# **Answer:** `university_tier` has the largest mediated effect — 61.5% of
# its total effect on hiring flows through the AI bias score, similar in
# proportion to gender (60.0% mediated). `education` barely mediates at all
# (13.8%), which fits with Step 5 showing almost no raw hiring gap by
# education to begin with. `age` is the odd one out: its direct and
# indirect effects point in *opposite* directions (direct = -0.169,
# indirect = +0.041), which is a real anomaly worth flagging rather than
# glossing over — it suggests age's relationship with hiring and the AI
# score isn't a simple "more bias flows through the score" story the way
# it is for gender and university tier.
#
# The refutation test on `university_tier` is reassuring: when I swap in a
# fake, randomly-generated treatment instead of the real one, the estimated
# effect drops from -0.128 to essentially zero (-0.0009, p = 0.90). That's
# what should happen if my original estimate reflects a real relationship
# and not just an artifact of the model. This gives me more confidence that
# the university tier finding is trustworthy, not a fluke of how the
# confounders happened to be structured.

# %% [markdown]
# ---
# ## Question 3: At the individual level, how many boundary candidates would flip?
#
# My plan: build one model predicting `hired` from all the qualification
# variables, the AI bias score, and the protected attribute variables.
# Find candidates whose predicted probability of being hired is close to
# 0.5 — these are the people where the decision could easily have gone
# either way. Then simulate what would happen to their prediction if their
# AI bias score had been the dataset's average instead of their actual
# score, holding everything else about them fixed.

# %% [markdown]
# ## Step 16: Fit a full model predicting hired

# %%
# Set up the features: qualifications, AI bias score, and protected attributes
full_features = qualification_vars + ['ai_bias_score', 'is_female', 'is_non_binary', 'university_tier_num', 'is_bachelors', 'age']

X_full = hiring_df[full_features]
y_full = hiring_df['hired']

full_model = LogisticRegression(max_iter=1000, random_state=1)
full_model.fit(X_full, y_full)

# %% [markdown]
# ## Step 17: Get each candidate's predicted probability of being hired

# %%
# predict_proba returns two columns: probability of 0, probability of 1
# I only need the probability of being hired (column index 1)
hiring_df['predicted_prob'] = full_model.predict_proba(X_full)[:, 1]

# Review a few predictions
hiring_df[['candidate_id', 'hired', 'predicted_prob']].head()

# %% [markdown]
# ## Step 18: Identify the boundary candidates
#
# I'm defining "boundary" as a predicted probability between 0.45 and 0.55
# — close enough to the 0.5 cutoff that the AI score could plausibly be
# the deciding factor.

# %%
# Filter to candidates near the decision boundary
boundary_candidates = hiring_df[
    (hiring_df['predicted_prob'] >= 0.45) & (hiring_df['predicted_prob'] <= 0.55)
].copy()

print(f"Number of boundary candidates: {len(boundary_candidates)}")

# %% [markdown]
# ## Step 19: Build the counterfactual scenario
#
# For just the boundary candidates, I'm replacing their actual
# `ai_bias_score` with the dataset's average score, keeping everything
# else about them exactly the same, and re-predicting.

# %%
# Calculate the dataset's average AI bias score
average_bias_score = hiring_df['ai_bias_score'].mean()
print(f"Average AI bias score across all candidates: {round(average_bias_score, 2)}")

# %%
# Build the counterfactual feature set: same as actual, but with the average bias score
X_counterfactual = boundary_candidates[full_features].copy()
X_counterfactual['ai_bias_score'] = average_bias_score

# Get counterfactual predicted probabilities
boundary_candidates['counterfactual_prob'] = full_model.predict_proba(X_counterfactual)[:, 1]

# %% [markdown]
# ## Step 20: Check how many candidates would flip outcome

# %%
# Convert both probabilities into a hire/no-hire decision using a 0.5 cutoff
boundary_candidates['actual_decision'] = (boundary_candidates['predicted_prob'] >= 0.5).astype(int)
boundary_candidates['counterfactual_decision'] = (boundary_candidates['counterfactual_prob'] >= 0.5).astype(int)

# Candidates whose decision changes between the actual and counterfactual scenario
flipped_candidates = boundary_candidates[
    boundary_candidates['actual_decision'] != boundary_candidates['counterfactual_decision']
]

print(f"Boundary candidates: {len(boundary_candidates)}")
print(f"Candidates whose outcome flips if their AI score were average: {len(flipped_candidates)}")
print(f"Percentage: {round(100 * len(flipped_candidates) / len(boundary_candidates), 1)}%")

# %% [markdown]
# ## Step 21: Look at who those flipped candidates are
#
# I want to see whether the flipped candidates skew toward any particular
# protected attribute group, which would tie this individual-level finding
# back to the aggregate patterns from Questions 1 and 2.

# %%
# Review the protected attributes of the candidates whose outcome flipped
flipped_candidates[['candidate_id', 'gender', 'university_tier', 'predicted_prob', 'counterfactual_prob', 'ai_bias_score']].head(10)

# %%
# Check the gender breakdown of flipped candidates vs. all boundary candidates
print("Gender breakdown, all boundary candidates:")
print(boundary_candidates['gender'].value_counts(normalize=True).round(3))
print()
print("Gender breakdown, flipped candidates only:")
print(flipped_candidates['gender'].value_counts(normalize=True).round(3))

# %% [markdown]
# **Answer:** Out of 249 candidates sitting right at the hire/no-hire
# boundary, 111 (44.6%) would have gotten a different outcome if their AI
# bias score had simply been the dataset average instead of their actual
# score — holding every other part of their application exactly the same.
# For nearly half of the candidates in the gray zone, the AI score alone is
# the deciding factor.
#
# One nuance worth being honest about: when I checked whether the flipped
# candidates skew toward any particular gender or university tier compared
# to the boundary group overall, they don't — the breakdowns are nearly
# identical (e.g. Female candidates are 43.8% of the boundary group and
# 45.0% of the flipped group). So while Questions 1 and 2 show the AI score
# is systematically influenced by gender and university tier at the
# aggregate level, this individual-level cut doesn't show the flips
# themselves concentrated in one demographic group. The mechanism (the AI
# score swinging outcomes for boundary candidates) is real and large, but
# at this specific boundary-cases cut, it isn't visibly concentrated by
# group — that's a limitation of this particular slice of the analysis,
# not a contradiction of the earlier findings.
