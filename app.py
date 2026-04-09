from flask import Flask, render_template, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

# load the trained model and supporting files
base_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_dir, 'pl_model.pkl'))
feature_list = joblib.load(os.path.join(base_dir, 'pl_features.pkl'))
label_encoders = joblib.load(os.path.join(base_dir, 'pl_encoders.pkl'))


def get_rate(cibil, emp_type):
    # interest rate based on cibil score band - standard bank practice
    if cibil >= 800:
        rate = 10.5
    elif cibil >= 750:
        rate = 11.5
    elif cibil >= 700:
        rate = 13.0
    elif cibil >= 650:
        rate = 15.0
    else:
        rate = 17.5
    if emp_type == 'Self-Employed':
        rate += 1.0
    return round(rate, 1)


def calc_emi(principal, annual_rate, months):
    # standard EMI formula: P * r * (1+r)^n / ((1+r)^n - 1)
    r = annual_rate / 100 / 12
    if r == 0:
        return round(principal / months)
    emi = principal * r * (1 + r) ** months / ((1 + r) ** months - 1)
    return round(emi)


def get_max_loan(income, cibil, existing_emi, expenses, interest_rate, tenure):
    # step 1: what EMI can this person actually afford?
    # banks allow max 40% of income as total EMI (existing + new)
    max_emi_capacity = (income * 0.40) - existing_emi

    # step 2: also check if they have enough disposable income
    disposable = income - expenses - existing_emi
    disposable_cap = disposable * 0.30  # only 30% of what's left after expenses

    # take the safer (lower) number
    affordable_emi = max(0, min(max_emi_capacity, disposable_cap))

    if affordable_emi <= 0:
        return 0

    # step 3: back-calculate loan amount from that affordable EMI
    r = interest_rate / 100 / 12
    if r == 0:
        loan_from_emi = affordable_emi * tenure
    else:
        loan_from_emi = affordable_emi * ((1 + r) ** tenure - 1) / (r * (1 + r) ** tenure)

    # step 4: cibil score also puts a ceiling on multiplier
    if cibil >= 750:
        cibil_cap = income * 20
    elif cibil >= 700:
        cibil_cap = income * 15
    elif cibil >= 650:
        cibil_cap = income * 10
    else:
        cibil_cap = income * 6

    # final answer: whichever is lower - EMI limit or CIBIL limit
    max_loan = min(loan_from_emi, cibil_cap)
    return max(0, int(round(max_loan / 5000) * 5000))


def build_feedback(income, dti, cibil, loan_amt, work_exp, emp_type, emp_cat, education, existing_emi, expenses):
    strengths = []
    problems = []
    tips = []

    loan_to_income = loan_amt / income if income > 0 else 99

    # cibil
    if cibil >= 750:
        strengths.append(f"Excellent CIBIL score ({cibil}) — strongly favours approval")
    elif cibil >= 700:
        strengths.append(f"Good CIBIL score ({cibil})")
    elif cibil >= 650:
        problems.append(f"Average CIBIL score ({cibil}) — lenders prefer 700+")
        tips.append("Pay all EMIs and credit card bills on time for 6-12 months to push CIBIL above 700")
    else:
        problems.append(f"Low CIBIL score ({cibil}) — banks see this as high default risk")
        tips.append("Check your CIBIL report for errors. Clear any overdue loans first")

    # dti ratio
    if dti < 0.35:
        strengths.append(f"Healthy debt-to-income ratio ({dti:.0%}) — good repayment capacity")
    elif dti < 0.50:
        strengths.append(f"Acceptable DTI ({dti:.0%})")
    elif dti < 0.65:
        problems.append(f"High DTI ({dti:.0%}) — over 50% of income is going to obligations")
        tips.append("Close one existing EMI before applying — it will bring DTI below 50%")
    else:
        problems.append(f"Very high DTI ({dti:.0%}) — almost no buffer left for new repayment")
        tips.append("Clear at least one existing loan before reapplying")

    # income
    if income >= 100000:
        strengths.append(f"Strong monthly income (₹{income:,.0f}) — good repayment ability")
    elif income >= 50000:
        strengths.append(f"Decent income (₹{income:,.0f})")
    elif income >= 25000:
        problems.append(f"Moderate income (₹{income:,.0f}) — limits how much you can borrow")
        tips.append("Adding a co-applicant (spouse or parent) with income can increase your eligible amount")
    else:
        problems.append(f"Low income (₹{income:,.0f}) — significantly restricts eligibility")
        tips.append("Consider a smaller loan or a secured loan against FD or property")

    # loan to income ratio
    if loan_to_income < 5:
        strengths.append(f"Requested amount is {loan_to_income:.1f}x monthly income — very manageable")
    elif loan_to_income >= 15:
        problems.append(f"Requested amount is {loan_to_income:.1f}x monthly income — too high for your income")
        tips.append("Reduce the loan amount or increase tenure to lower the EMI burden")
    elif loan_to_income >= 10:
        problems.append(f"Requested amount is {loan_to_income:.1f}x monthly income — on the higher side")
        tips.append("Request a lower loan amount or longer tenure")

    # employer
    if emp_cat in ['Government', 'MNC']:
        strengths.append(f"{emp_cat} employee — lenders trust this income stability")
    elif emp_cat == 'Startup':
        problems.append("Startup employment is seen as higher risk by lenders")
        tips.append("Switching to a more stable company will improve your chances")

    if emp_type == 'Self-Employed':
        tips.append("File ITR for at least 2 years — lenders need proof of stable self-employment income")

    # experience
    if work_exp >= 5:
        strengths.append(f"{work_exp} years of experience shows career stability")
    elif work_exp < 2:
        problems.append("Less than 2 years experience — lenders prefer a stable job history")
        tips.append("Complete 2 years in your current role before reapplying")

    # education (small factor)
    if education in ['Master', 'PhD']:
        strengths.append(f"{education}'s degree suggests higher earning potential")

    return strengths, problems, tips


@app.route('/')
def home():
    return "Loan Eligibility Predictor is running!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()

        income = float(data['Net_Monthly_Income'])
        expenses = float(data['Monthly_Expenses'])
        existing_emi = float(data.get('Existing_EMI', 0) or 0)
        cibil = int(data['CIBIL_Score'])
        loan_amt = float(data['Loan_Amount_Requested'])
        tenure = int(data['Loan_Tenure_Months'])
        emp_type = data['Employment_Type']
        emp_cat = data['Employer_Category']
        education = data['Education']
        work_exp = int(data['Work_Experience_Yrs'])

        dti = round((expenses + existing_emi) / income, 3) if income > 0 else 0.99

        row = {
            'Age': int(data['Age']),
            'Gender': label_encoders['Gender'][data['Gender']],
            'Marital_Status': label_encoders['Marital_Status'][data['Marital_Status']],
            'Dependents': int(data['Dependents']),
            'Education': label_encoders['Education'][education],
            'Employment_Type': label_encoders['Employment_Type'][emp_type],
            'Employer_Category': label_encoders['Employer_Category'][emp_cat],
            'Work_Experience_Yrs': work_exp,
            'City_Tier': label_encoders['City_Tier'][data['City_Tier']],
            'Net_Monthly_Income': income,
            'Monthly_Expenses': expenses,
            'Existing_EMI': existing_emi,
            'Debt_To_Income_Ratio': dti,
            'CIBIL_Score': cibil,
            'Loan_Amount_Requested': loan_amt,
            'Loan_Tenure_Months': tenure,
        }

        input_df = pd.DataFrame([row], columns=feature_list)
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = round(float(max(probabilities)) * 100, 1)
        approved = bool(prediction == 1)

        interest_rate = get_rate(cibil, emp_type)
        max_loan = get_max_loan(income, cibil, existing_emi, expenses, interest_rate, tenure)
        emi_on_requested = calc_emi(loan_amt, interest_rate, tenure)
        emi_on_max = calc_emi(max_loan, interest_rate, tenure) if max_loan > 0 else 0

        strengths, problems, tips = build_feedback(
            income, dti, cibil, loan_amt, work_exp,
            emp_type, emp_cat, education, existing_emi, expenses
        )

        return jsonify({
            'approved': approved,
            'confidence': confidence,
            'dti': round(dti * 100, 1),
            'interest_rate': interest_rate,
            'max_eligible': max_loan,
            'emi_on_requested': emi_on_requested,
            'emi_on_max': emi_on_max,
            'strengths': strengths,
            'reasons': problems,
            'tips': tips,
        })

    except KeyError as e:
        return jsonify({'error': f'Missing or invalid field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'GradientBoostingClassifier'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
