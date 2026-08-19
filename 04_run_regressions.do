version 17.0
clear all
set more off
set seed 20260819

/*
Section 1 regressions for the DP chain-of-thought analysis.

Run from the project directory after creating pilot_codebook.csv:
    stata -b do 04_run_regressions.do

Outputs are written to stata_results/. Regression (1) is a separate
mixed-effects logit for every binary code. Regression (2) is estimated as:
  a. the specified score model with F, P, C, all codes, and crossed random
     intercepts for model and product;
  b. the preliminary OLS score model using all 19 codes; and
  c. cross-validated elastic net using all 19 codes.
*/

local pilot_file    "pilot1_calls.csv"
local codebook_file "pilot_codebook.csv"
local results_dir   "stata_results"

capture mkdir "`results_dir'"
capture log close _all
log using "`results_dir'/section1_regressions.log", text replace name(section1)

capture confirm file "`pilot_file'"
if _rc {
    display as error "Missing input: `pilot_file'"
    exit 601
}
capture confirm file "`codebook_file'"
if _rc {
    display as error "Missing input: `codebook_file'"
    exit 601
}

local codes ///
    mentions_product_attributes ///
    mentions_activity_message ///
    mentions_rating ///
    mentions_testimonial ///
    mentions_assurance ///
    uptake_activity_message ///
    uptake_rating ///
    uptake_testimonial ///
    uptake_assurance ///
    questions_claim_validity ///
    recognizes_persuasion ///
    considers_price_value ///
    considers_suitability ///
    considers_risk ///
    changes_original_decision ///
    evaluates_product_features ///
    performs_comparison ///
    explicit_character ///
    questions_branding

/* Read and validate the eligible pilot rows. */
import delimited using "`pilot_file'", varnames(1) clear encoding(utf8)
foreach required in response_id response_status parse_status F P C m j S {
    capture confirm variable `required'
    if _rc {
        display as error "pilot1_calls.csv is missing variable: `required'"
        exit 111
    }
}
keep if lower(strtrim(response_status)) == "ok" & lower(strtrim(parse_status)) == "ok"
isid response_id

capture confirm numeric variable F
if _rc destring F, replace
capture confirm numeric variable C
if _rc destring C, replace
capture confirm numeric variable S
if _rc destring S, replace
assert inlist(F, 0, 1)
assert inlist(C, 0, 1)
assert !missing(S)

encode P, gen(pattern_id)
encode m, gen(model_id)
encode j, gen(product_id)
quietly levelsof pattern_id if lower(strtrim(P)) == "control", local(control_level)
local control_count : word count `control_level'
if `control_count' != 1 {
    display as error "P must contain exactly one encoded control level."
    exit 459
}
fvset base `control_level' pattern_id
label variable pattern_id "Social-proof treatment (control is base)"
label variable model_id "Model"
label variable product_id "Product"

tempfile pilot
save `pilot'

/* Read the LLM-coded outcomes and merge one-to-one by globally unique ID. */
import delimited using "`codebook_file'", varnames(1) clear encoding(utf8)
capture confirm variable response_id
if _rc {
    display as error "pilot_codebook.csv is missing variable: response_id"
    exit 111
}
isid response_id
foreach code of local codes {
    capture confirm variable `code'
    if _rc {
        display as error "pilot_codebook.csv is missing code: `code'"
        exit 111
    }
    capture confirm numeric variable `code'
    if _rc destring `code', replace
    assert inlist(`code', 0, 1)
}
keep response_id `codes'
merge 1:1 response_id using `pilot', keep(match) nogen

quietly count
local analysis_n = r(N)
display as result "Merged analysis sample: `analysis_n' observations"
if `analysis_n' == 0 {
    display as error "No matched observations after merging the two CSV files."
    exit 2000
}

/*
Regression (1):
  logit Pr(M_ik=1) = a0 + aF*F + sum_p aPp*Pp + aC*C + b_model + u_product

The first random-effects equation plus the product grouping equation specify
crossed (not nested) random intercepts. Product is last for efficient fitting.
*/
local code_number = 0
foreach code of local codes {
    local ++code_number
    display as text _newline "Regression (1), outcome: `code'"
    quietly summarize `code'
    if r(min) == r(max) {
        display as error "Skipped `code': the outcome has no variation."
        continue
    }
    capture noisily melogit `code' i.F i.pattern_id i.C ///
        || _all: R.model_id || product_id:, or nolog
    if _rc {
        display as error "melogit failed for `code' with return code " _rc
        continue
    }
    local estimate_name = "logit" + string(`code_number', "%02.0f")
    estimates store `estimate_name'
    estimates save "`results_dir'/logit_`code'.ster", replace
}

/*
Regression (2a), robust specification from the PDF:
  S = b0 + bF*F + sum_p bPp*Pp + bC*C + sum_k gamma_k*M_ik
      + b_model + u_product + error
*/
display as text _newline "Regression (2a): full mixed-effects score model"
capture noisily mixed S i.F i.pattern_id i.C `codes' ///
    || _all: R.model_id || product_id:, vce(robust) nolog
if _rc {
    display as error "Full mixed-effects score regression failed with return code " _rc
}
else {
    estimates store score_mixed
    estimates save "`results_dir'/score_mixed.ster", replace
}

/* Regression (2b), simpler preliminary score model in the PDF. */
display as text _newline "Regression (2b): preliminary 19-code OLS model"
regress S `codes', vce(robust)
estimates store score_ols
estimates save "`results_dir'/score_ols.ster", replace

/*
Regression (2c): elastic net with 10-fold cross-validation to select alpha
(mixing) and lambda (penalty). This is predictive regularization and does not
replace the inferential mixed/OLS specifications above.
*/
display as text _newline "Regression (2c): cross-validated elastic net"
capture noisily elasticnet linear S `codes', selection(cv, folds(10)) ///
    rseed(20260819) nolog
if _rc {
    display as error "Elastic net failed or is unavailable; return code " _rc
}
else {
    estimates store score_elasticnet
    estimates save "`results_dir'/score_elasticnet.ster", replace
    lassocoef, display(coef, standardized)
}

compress
save "`results_dir'/section1_analysis_data.dta", replace
log close section1
display as result "Section 1 analysis complete. See `results_dir'/ for outputs."
