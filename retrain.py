"""
PhishDetect AI — Model Retraining Script
Run: python3 retrain.py

Completely replaces the Enron-biased models with a properly balanced dataset
covering diverse legitimate emails, modern phishing, and AI-generated phishing.

Saves: model.pkl, vectorizer.pkl, svm_final.pkl
"""

import os, re, html, random, warnings, json
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
LABELS = {0: "legitimate", 1: "traditional phishing", 2: "ai-generated phishing"}

# ══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING — matches phishdetect_util.py (keep in sync)
# ══════════════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url_link ", text)
    text = re.sub(r"\S+@\S+\.\S+", " email_addr ", text)
    text = re.sub(r"\b\d[\d,.]*\b", " number ", text)
    text = re.sub(r"[^a-z\s_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA — covers every failure case we've encountered
# ══════════════════════════════════════════════════════════════════════════════

def _vary(templates: list[str], n: int) -> list[str]:
    """Return n samples drawn (with replacement) from a template list."""
    return [random.choice(templates) for _ in range(n)]

# ── Slot values for template interpolation ────────────────────────────────────
_NAMES       = ["David", "Emeka", "Fatima", "Ngozi", "Tunde", "Ibrahim", "Chukwudi",
                "Sarah", "James", "Aisha", "Michael", "Blessing", "John", "Amaka"]
_BANKS_NG    = ["GTBank", "Zenith Bank", "Access Bank", "First Bank", "UBA",
                "Fidelity Bank", "FCMB", "Stanbic IBTC", "Sterling Bank", "Wema Bank",
                "Kuda Bank", "Moniepoint"]
_BANKS_INTL  = ["Barclays", "HSBC", "Lloyds Bank", "Bank of America", "Chase",
                "Wells Fargo", "Citibank", "Deutsche Bank", "Standard Chartered"]
_AMOUNTS_NG  = ["5,000", "12,500", "50,000", "150,000", "2,500", "75,000",
                "200,000", "1,500", "3,750", "25,000", "500,000"]
_AMOUNTS_USD = ["49.99", "150.00", "299.00", "1,200.00", "89.95", "24.99",
                "599.00", "12.99", "399.99"]
_DATES       = ["June 10, 2026", "June 11, 2026", "May 28, 2026",
                "April 15, 2026", "March 3, 2026", "07/06/2026"]
_LAST4       = ["4521", "7890", "3312", "6601", "9045", "2278"]
_PHONES      = ["0700-123-4567", "08012345678", "+234 901 234 5678", "01-2806800"]
_REFS        = ["TXN-2026-88234", "REF/2026/00123", "INV-98745", "ORD-20260611-445"]

def _fill(template: str) -> str:
    return (template
        .replace("{name}",       random.choice(_NAMES))
        .replace("{bank}",       random.choice(_BANKS_NG))
        .replace("{intl_bank}",  random.choice(_BANKS_INTL))
        .replace("{amount_ng}",  random.choice(_AMOUNTS_NG))
        .replace("{amount_usd}", random.choice(_AMOUNTS_USD))
        .replace("{date}",       random.choice(_DATES))
        .replace("{last4}",      random.choice(_LAST4))
        .replace("{phone}",      random.choice(_PHONES))
        .replace("{ref}",        random.choice(_REFS))
    )

def _fill_all(templates: list[str], n: int) -> list[str]:
    base = [_fill(t) for t in templates]
    out  = []
    while len(out) < n:
        out.extend([_fill(t) for t in templates])
    return out[:n]


# ── CLASS 0: LEGITIMATE ───────────────────────────────────────────────────────

LEGIT_BANK_ALERTS = [
    "Dear {name}, your {bank} account ending {last4} has been debited NGN {amount_ng} on {date}. Transaction reference: {ref}. Available balance: NGN {amount_ng}. If you did not authorise this, call {phone} immediately.",
    "{bank} Debit Alert. Amount: NGN {amount_ng}. Account: ****{last4}. Date: {date}. Description: POS Purchase. Balance: NGN {amount_ng}. Queries? Call {phone} or visit any {bank} branch.",
    "CREDIT ALERT — {bank}. Dear {name}, your account ****{last4} has been credited with NGN {amount_ng} on {date}. Ref: {ref}. New balance: NGN {amount_ng}. Thank you for banking with us.",
    "Transaction Notification from {bank}. Your account was used for a web transfer of NGN {amount_ng} on {date}. If authorised, no action is required. For concerns, contact {phone}.",
    "Your {bank} account statement for May 2026 is now available. Log in to internet banking to view your transactions. Account ****{last4}. Contact us: {phone}.",
    "Dear {name}, we have received your transfer request of NGN {amount_ng} to beneficiary account on {date}. Reference: {ref}. The transaction will be processed within 24 hours.",
    "{bank} Security Notice: A new device has been linked to your account on {date}. If this was you, no action is needed. If not, call {phone} immediately to secure your account.",
    "Dear {name}, your {bank} loan instalment of NGN {amount_ng} was successfully debited on {date}. Remaining balance: NGN {amount_ng}. Thank you.",
]

LEGIT_ECOMMERCE = [
    "Your eBay order has been confirmed. Item: Sony WH-1000XM5 Headphones. Order number: {ref}. Estimated delivery: {date}. Total charged: ${amount_usd}. Track your order in My eBay.",
    "Hello {name}, thank you for your purchase! Your order {ref} has been shipped. Expected delivery: {date}. You can track your package using the link in your account.",
    "Your Amazon order {ref} is on its way. We'll send you another email when your package is delivered. Estimated arrival: {date}.",
    "Order Confirmation — Jumia. Hi {name}, we've received your order {ref} for NGN {amount_ng}. Your items will be delivered by {date}. Track your delivery on the Jumia app.",
    "Receipt from Apple. Amount billed: ${amount_usd}. Date: {date}. Item: iCloud+ 50GB. Apple ID: email_addr. Questions? Visit Apple Support or call {phone}.",
    "Your Konga order is confirmed! Order ID: {ref}. Total: NGN {amount_ng}. Delivery address: Lekki, Lagos. Estimated delivery: {date}.",
    "Dear {name}, your return request for order {ref} has been approved. Refund of ${amount_usd} will be processed to your original payment method within 5-7 business days.",
    "eBay Purchase Notification. {name}, you've just bought: Laptop Bag - Black. Total: ${amount_usd}. Payment received {date}. View purchase details in your account.",
]

LEGIT_FINTECH = [
    "Paystack Payment Notification. Hello {name}, a payment of NGN {amount_ng} was successfully received by your business on {date}. Reference: {ref}. View transaction details in your Paystack dashboard.",
    "Your Flutterwave payout of NGN {amount_ng} has been initiated. Reference: {ref}. Funds will be credited to your bank account within 24 hours. Date: {date}.",
    "Kuda Bank Alert. Your wallet has been credited with NGN {amount_ng} from {name}. New balance: NGN {amount_ng}. {date}.",
    "OPay Notification. Dear {name}, you have received a transfer of NGN {amount_ng} from a contact. Your new OPay balance is NGN {amount_ng}.",
    "PiggyVest Savings Alert. Hi {name}, your automatic savings of NGN {amount_ng} was successfully locked on {date}. Total savings: NGN {amount_ng}. Keep it up!",
    "Moniepoint POS Receipt. Terminal: {ref}. Amount: NGN {amount_ng}. Date: {date}. Transaction approved. Merchant copy. Thank you for your business.",
    "Your Wise transfer of {amount_usd} USD has been sent. Recipient will receive the funds within 1-2 business days. Reference: {ref}.",
]

LEGIT_FORMAL_NOTICE = [
    "Important Notice: Temporary Office Closure. Dear Valued Customer, in observance of Democracy Day, our offices will be closed on Friday June 13, 2026. Services remain available online and via our mobile app. Contact {phone} for urgent matters.",
    "Dear {name}, this is to inform you that due to scheduled maintenance, our internet banking platform will be unavailable from 12:00 AM to 4:00 AM on {date}. We apologise for any inconvenience.",
    "Annual General Meeting Notice. Dear Shareholder, you are cordially invited to attend the Annual General Meeting of the company on {date}. Please confirm your attendance by replying to this email.",
    "Policy Update Notification. Dear {name}, we have updated our Terms and Conditions effective {date}. The key changes relate to data privacy and account management. Please review the updated policy on our website.",
    "Maintenance Notification. Our systems will undergo planned maintenance on {date} from 2:00 AM to 6:00 AM WAT. During this period, some services may be temporarily unavailable. We apologise for any disruption.",
    "Dear Customer, following the recent amendment to the Central Bank of Nigeria guidelines, we wish to inform you that there have been updates to transaction limits effective {date}. For details, please visit any of our branches.",
    "NOTICE TO ALL STAFF: The office will be closed on {date} in observance of the public holiday declared by the Federal Government. Normal operations resume the following working day.",
]

LEGIT_SAAS_NOTIFICATIONS = [
    "CHAOSS Community (via Slack). Your team has sent 181 messages recently. 1 more teammate has joined. View all unread messages in your Slack workspace.",
    "GitHub notification: {name} opened a new pull request in chaoss/augur. PR #445: Add new metric endpoint. Review it at github.com/chaoss/augur/pull/445",
    "Zoom: Your meeting 'Weekly Standup' starts in 15 minutes. Join at zoom.us/j/{ref}. Meeting ID: 812 456 7890.",
    "Notion: {name} invited you to collaborate on 'Q2 Project Plan'. Click the link in your Notion app to accept and start collaborating.",
    "Google Calendar reminder: Product Review Meeting starts at 3:00 PM today. Organiser: {name}. Location: Conference Room B.",
    "Your Figma file 'App Redesign v3' was shared with you by {name}. Open Figma to view and comment.",
    "Jira: Issue PROJ-234 has been assigned to you by {name}. Summary: Fix login timeout bug. Priority: High. Due: {date}.",
    "Trello: {name} added you to the card 'Deploy to production' in the board 'Sprint 14'. View the card to see the latest updates.",
    "New comment on your GitHub pull request. {name} commented: 'Looks good, just fix the linting error on line 42.' View the discussion at github.com.",
]

LEGIT_PENSION_INSURANCE = [
    "Dear {name}, your Leadway Pensure PFA monthly contribution of NGN {amount_ng} has been received and credited to your Retirement Savings Account for {date}. Your RSA balance is NGN {amount_ng}.",
    "ARM Pension Managers: Dear {name}, your RSA statement for Q1 2026 is now available. Log in to the ARM Pension portal to download your statement. For enquiries, call {phone}.",
    "Stanbic IBTC Pension Administrators. Dear {name}, please be informed that your employer has remitted your pension contribution of NGN {amount_ng} for the month. Total RSA balance: NGN {amount_ng}.",
    "AXA Mansard Insurance. Policy Renewal Notice. Dear {name}, your motor insurance policy {ref} expires on {date}. Renewal premium: NGN {amount_ng}. Contact us on {phone} to renew.",
    "AIICO Insurance: Dear {name}, your life insurance premium of NGN {amount_ng} has been successfully debited from your account. Policy reference: {ref}. Thank you.",
    "Hygeia HMO: Your health maintenance subscription has been renewed for the period {date}. Your HMO ID: {ref}. Contact {phone} for any medical emergencies.",
]

LEGIT_UTILITY_GOVT = [
    "EKEDC (Eko Electricity Distribution Company). Dear {name}, your electricity bill for {date} is NGN {amount_ng}. Account number: {ref}. Pay before due date to avoid disconnection. Pay via bank transfer, USSD, or visit any payment centre.",
    "Lagos State Internal Revenue Service (LIRS). Dear {name}, your personal income tax assessment for 2025 has been computed. Tax payable: NGN {amount_ng}. Payment deadline: {date}. Reference: {ref}.",
    "LAWMA Notice: Your waste management levy for {date} is now due. Amount: NGN {amount_ng}. Pay at any commercial bank using reference {ref}.",
    "Federal Inland Revenue Service (FIRS). Your TIN: {ref}. Your VAT filing for {date} has been received. Amount: NGN {amount_ng}. This is your electronic receipt.",
    "NIMC Notification: Your National Identification Number (NIN) enrolment has been completed. Your NIN is: {ref}. This information is confidential. Do not share with third parties.",
]

LEGIT_HR_CORPORATE = [
    "Payroll Notification: Dear {name}, your salary of NGN {amount_ng} for the month of May 2026 has been processed and will be credited to your account by {date}. Reference: {ref}.",
    "HR Notice: Please note that the deadline for submission of your 2025 annual leave application is {date}. Submit your application through the HR portal before this date.",
    "Meeting Invitation: You are invited to the Q2 Performance Review meeting on {date} at 10:00 AM. Location: Boardroom 3. Please confirm your attendance by replying to this email.",
    "IT Department: Scheduled system maintenance will occur on {date}. Please save all work and log out of company systems by 11:00 PM. Normal operations will resume by 6:00 AM.",
    "Dear {name}, your employment confirmation letter is attached to this email. Please sign and return the attached document by {date}. For questions, contact HR on {phone}.",
    "Training Announcement: A mandatory cybersecurity awareness training session has been scheduled for {date}. All staff are required to attend. Venue: Training Hall, 3rd Floor.",
]

# ── CLASS 1: TRADITIONAL PHISHING ─────────────────────────────────────────────

PHISH_ACCOUNT_THREATS = [
    "URGENT: Your account has been suspended. Dear User, we have detected unusual activity on your account. Your account has been temporarily suspended. Click the link below to verify your identity within 24 hours or your account will be permanently closed.",
    "IMPORTANT NOTICE: Your online banking access has been blocked due to multiple failed login attempts. You must verify your account immediately by clicking here. Failure to do so within 12 hours will result in account termination.",
    "Alert: Suspicious login detected on your account from an unrecognised device in Nigeria. If this was not you, click here immediately to secure your account and change your password.",
    "Your account is at risk! We have flagged your account for suspicious activity. To avoid suspension, please verify your personal details by following the link. Act now before your account is permanently closed.",
    "Final Warning: Your account will be closed in 48 hours. To prevent this, you must complete verification by clicking the link and entering your username, password, and date of birth.",
]

PHISH_PRIZE_LOTTERY = [
    "CONGRATULATIONS! You have been selected as a winner in our international lottery! You have won $1,500,000. To claim your prize, send your full name, address, and bank account details to our agent immediately.",
    "You are a winner! Your email address was selected in our monthly draw. Prize: $750,000 USD. To receive your winnings, you must pay a processing fee of $250 to release the funds. Contact our claims office urgently.",
    "WINNER NOTIFICATION: Google International has selected you as one of five lucky winners of the 2026 Google Annual Promotion. You have won a laptop and $50,000 cash. Reply with your details to claim.",
    "CLAIM YOUR PRIZE IMMEDIATELY! Your mobile number was randomly selected by MTN Nigeria for a cash prize of NGN 500,000. To claim, send your name and account number. Hurry — offer expires in 24 hours!",
]

PHISH_CREDENTIAL_HARVEST = [
    "Dear Customer, your internet banking password will expire today. You must update your password immediately by clicking the link below. Enter your current username and password to continue.",
    "Security Alert from your bank: Your debit card has been flagged for fraudulent transactions. To prevent further unauthorised charges, please verify your card details including card number, expiry date, and CVV.",
    "ACTION REQUIRED: Your PayPal account needs verification. We noticed your account information does not match our records. Please login to update your details. Click here to verify: paypa1-secure.net/verify",
    "Dear valued customer, to continue using our online services, you are required to re-validate your personal information. Please provide your date of birth, mother's maiden name, and account PIN via the link.",
    "Your account has been limited. To restore full access, please verify your identity by providing your account number, sort code, and online banking login credentials through our secure verification page.",
]

PHISH_DELIVERY_SCAM = [
    "Your package could not be delivered. Delivery attempt was unsuccessful due to incomplete address. A delivery fee of $2.99 is required to reschedule. Click here to pay and schedule redelivery before your package is returned.",
    "DHL: Your shipment ref {ref} is on hold. Please pay the customs clearance fee of NGN 3,500 to release your package. Failure to pay within 48 hours will result in return of the package.",
    "FedEx Delivery Notification: Your parcel has arrived at our facility but requires additional information to complete delivery. Please click here and provide your updated address and a small insurance fee.",
    "NIPOST: Your registered mail is awaiting collection. Kindly pay a handling fee of NGN 1,500 online before visiting our office. Reference: {ref}.",
]

PHISH_INVESTMENT_ROMANCE = [
    "Hello dear, I found your profile online and I think we could be great friends. I am a doctor working with the UN. I have a lucrative investment opportunity that can earn you $5,000 weekly. Let me know if you are interested.",
    "INVESTMENT ALERT: Our proprietary trading algorithm has generated 450% returns this quarter. We are accepting 50 new investors only. Minimum investment: $500. Returns paid weekly. Act fast before spots are filled.",
    "My name is Barrister Emmanuel. I represent the estate of a deceased client who shares your surname. A sum of $12.5 million is unclaimed. As next of kin, you are entitled to 40%. Reply for details.",
    "Emergency assistance needed. I am stranded abroad and my wallet was stolen. I need $800 to get home and will repay you immediately upon return. Please send via Western Union. I will explain everything later.",
]

PHISH_CEO_FRAUD = [
    "From the desk of the CEO. This is urgent and confidential. I need you to process a wire transfer of $45,000 to a vendor immediately. Do not discuss with anyone. I will explain the details later. Complete this before 5 PM.",
    "Hi, this is your MD. I'm in a meeting and need an urgent favour. Please purchase 10 x $100 iTunes gift cards and send me the codes. I will reimburse you. Keep this between us for now.",
    "Sensitive: Finance team, kindly process payment of $28,000 to the following account immediately. This is a board-approved transaction. Reference: {ref}. Do not delay. Contact me only by email for now.",
]

PHISH_BANK_IMPERSONATION = [
    "GTBANK URGENT ALERT: Your account has been compromised. Dear Customer, an unauthorised withdrawal of NGN 150,000 was attempted. Verify your BVN and PIN immediately at the link below to block this transaction.",
    "Access Bank Security: Your account login was attempted from Lagos and Abuja simultaneously. To prevent fraudulent access, click here and enter your ATM card number and PIN to lock your account.",
    "Zenith Bank: IMPORTANT — your account will be disabled due to incomplete BVN linkage. Kindly update your BVN and account details via the link. Failure to comply by {date} will freeze your funds.",
    "Your {intl_bank} account has been compromised. As a security measure, we have placed a temporary hold on your account. To restore access, log in using the secure link and confirm your identity with your card details.",
]

# ── CLASS 2: AI-GENERATED PHISHING ────────────────────────────────────────────

AI_PHISH_SOPHISTICATED = [
    "Dear {name}, I hope this message finds you well. I am writing on behalf of our compliance team regarding a routine review of your account activity. We have identified a discrepancy that requires your brief attention. To ensure your account remains in good standing, could you please take a moment to confirm a few details through our secure portal? The process takes under two minutes and will be completely resolved once verified. We appreciate your cooperation.",
    "Following our recent conversation with your organisation, I wanted to follow up on the invoice that appears to be outstanding. Please find attached the updated payment details for the amount of ${amount_usd}. Our finance team has indicated that payment is now overdue and we would appreciate settlement by {date} to avoid any disruption to services. Please click the link to access your payment portal.",
    "Good afternoon {name}, I am reaching out from the IT Security team regarding an upcoming mandatory password reset for all staff accounts. As part of our annual security audit, all passwords must be updated by {date}. Please use the link below to complete the process. The link will expire in 24 hours, so we recommend completing this at your earliest convenience.",
    "Hello {name}, thank you for your continued business with us. We are writing to let you know that your payment method on file will expire next month. To ensure uninterrupted service, please update your billing information at your earliest convenience using the secure link attached. This will only take a moment and will prevent any disruption to your account.",
    "Dear {name}, as part of our ongoing commitment to security, we are rolling out two-factor authentication for all user accounts effective {date}. To complete your enrollment, please click the link below and follow the on-screen instructions. If you do not complete setup within 48 hours, your access may be temporarily restricted.",
]

AI_PHISH_BEC = [
    "Hi {name}, hope you're doing well. Quick request — I'm currently in a meeting with a client and cannot take calls. I need to settle an invoice with one of our key vendors urgently. The amount is ${amount_usd}. Could you initiate a wire transfer to the account details below? I'll send you the PO and backup documents as soon as I'm out. This is time-sensitive so please prioritise. Thanks.",
    "Good morning, I need your help with something today. We've recently switched our payroll provider and I need to update my direct deposit information before the next pay cycle. Could you please update my bank account details on the system? New account: sort code 12-34-56, account number 87654321. I'd appreciate if this could be done quietly as it's a personal matter.",
    "I hope your week is going well. I wanted to reach out because I'll be travelling internationally starting tomorrow and won't have reliable access to my usual accounts. Could you process a payment on my behalf? The details are: beneficiary XYZ Consulting, amount ${amount_usd}, reference {ref}. I'll be available by email only. Much appreciated.",
]

AI_PHISH_SPEAR = [
    "Dear {name}, my name is Dr. Amanda Chen and I am a talent acquisition specialist at a leading investment bank. I came across your professional profile and was impressed by your experience. We have an opening that may be of interest. The position offers a base salary of $180,000 plus bonus. The first step would be a brief online assessment on our candidate portal. Could you complete this at your earliest convenience?",
    "Hi {name}, I am a researcher at Cambridge University studying fintech adoption in Nigeria. I found your name listed as an industry expert and would love your input for our study. The survey takes about 5 minutes and participants receive a $50 Amazon gift card. Please access the study at the link below using access code {ref}.",
    "Dear {name}, congratulations on your recent promotion. I noticed the announcement on LinkedIn and wanted to reach out. As your new role involves managing larger budgets, you may qualify for our corporate expense platform which offers significant advantages over personal cards. I've pre-filled an application for you based on your public profile. Just confirm your details to complete.",
]


# ══════════════════════════════════════════════════════════════════════════════
# BUILD DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_synthetic_dataset() -> tuple[list[str], list[int]]:
    texts, labels = [], []

    def add(templates, label, count):
        samples = _fill_all(templates, count)
        texts.extend(samples)
        labels.extend([label] * len(samples))

    # Legitimate (label 0)
    add(LEGIT_BANK_ALERTS,         0, 350)
    add(LEGIT_ECOMMERCE,           0, 300)
    add(LEGIT_FINTECH,             0, 250)
    add(LEGIT_FORMAL_NOTICE,       0, 250)
    add(LEGIT_SAAS_NOTIFICATIONS,  0, 250)
    add(LEGIT_PENSION_INSURANCE,   0, 200)
    add(LEGIT_UTILITY_GOVT,        0, 200)
    add(LEGIT_HR_CORPORATE,        0, 200)

    # Traditional phishing (label 1)
    add(PHISH_ACCOUNT_THREATS,     1, 350)
    add(PHISH_PRIZE_LOTTERY,       1, 300)
    add(PHISH_CREDENTIAL_HARVEST,  1, 300)
    add(PHISH_DELIVERY_SCAM,       1, 250)
    add(PHISH_INVESTMENT_ROMANCE,  1, 250)
    add(PHISH_CEO_FRAUD,           1, 200)
    add(PHISH_BANK_IMPERSONATION,  1, 300)

    # AI-generated phishing (label 2)
    add(AI_PHISH_SOPHISTICATED,    2, 500)
    add(AI_PHISH_BEC,              2, 350)
    add(AI_PHISH_SPEAR,            2, 350)

    return texts, labels


def try_download_real_data() -> tuple[list[str], list[int]]:
    """Try to pull additional real email data from HuggingFace. Returns empty lists on failure."""
    texts, labels = [], []
    try:
        from datasets import load_dataset
        print("Trying HuggingFace datasets...")

        sources = [
            ("talby/spam-ham-public-email-dataset", "email", "label", {"ham": 0, "spam": 1}),
            ("Deysi/spam-detection-dataset",        "text",  "label", {0: 0, 1: 1}),
        ]
        for ds_name, text_col, label_col, label_map in sources:
            try:
                ds = load_dataset(ds_name, trust_remote_code=True)
                split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
                for row in split:
                    t = str(row.get(text_col, "") or "")
                    l = row.get(label_col)
                    mapped = label_map.get(l, label_map.get(str(l)))
                    if t.strip() and mapped is not None:
                        texts.append(t)
                        labels.append(int(mapped))
                print(f"  ✓ {ds_name}: {len(texts)} rows")
                break
            except Exception as e:
                print(f"  ✗ {ds_name}: {e}")
    except ImportError:
        print("datasets library not available — using synthetic only.")
    return texts, labels


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train():
    print("\n" + "="*60)
    print("PhishDetect AI — Retraining")
    print("="*60)

    # 1. Build dataset
    print("\n[1/5] Building synthetic dataset...")
    texts, labels = build_synthetic_dataset()
    print(f"      Synthetic: {len(texts)} examples")

    extra_texts, extra_labels = try_download_real_data()
    if extra_texts:
        texts  += extra_texts
        labels += extra_labels
        print(f"      + Real data: {len(extra_texts)} examples")

    # Clean
    print("\n[2/5] Cleaning text...")
    cleaned = [clean_text(t) for t in texts]

    # Class counts
    from collections import Counter
    c = Counter(labels)
    print(f"      Class 0 (legitimate):          {c[0]}")
    print(f"      Class 1 (traditional phishing): {c[1]}")
    print(f"      Class 2 (AI phishing):          {c[2]}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.15, stratify=labels, random_state=42
    )

    # 3. Vectorise
    print("\n[3/5] Fitting TF-IDF vectorizer...")
    vec = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    print(f"      Vocabulary size: {len(vec.vocabulary_)}")

    # 4. Train LR
    print("\n[4/5] Training Logistic Regression...")
    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_tr, y_train)
    lr_pred = lr.predict(X_te)
    print(f"      LR accuracy: {accuracy_score(y_test, lr_pred):.3f}")

    # Train SVM
    print("      Training Calibrated SVM...")
    svm_base = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    svm = CalibratedClassifierCV(svm_base, cv=3)
    svm.fit(X_tr, y_train)
    svm_pred = svm.predict(X_te)
    print(f"      SVM accuracy: {accuracy_score(y_test, svm_pred):.3f}")

    # Ensemble
    lr_proba  = lr.predict_proba(X_te)
    svm_proba = svm.predict_proba(X_te)
    ens_proba = (lr_proba * 0.5) + (svm_proba * 0.5)
    ens_pred  = np.argmax(ens_proba, axis=1)
    ens_acc   = accuracy_score(y_test, ens_pred)
    print(f"      Ensemble accuracy: {ens_acc:.3f}")

    # 5. Evaluation
    print("\n[5/5] Evaluation")
    print("-"*60)
    print(classification_report(
        y_test, ens_pred,
        target_names=["Legitimate", "Traditional Phishing", "AI Phishing"],
        digits=3,
    ))

    # Specific test cases we care about
    print("\nSpot-check on known problem cases:")
    tests = [
        ("eBay order shipped",
         "Your eBay order has been confirmed and shipped. Order number 12345. Estimated delivery June 15. Total charged: $49.99.",
         0),
        ("Leadway Pensure pension notice",
         "Dear Valued Customer, in celebration of Democracy Day, the Federal Government has declared Friday June 13 2026 a public holiday. Our offices will be closed. The Pensure Online Platform remains available.",
         0),
        ("GTBank debit alert",
         "GTBank Debit Alert. Amount: NGN 50,000. Account: ****4521. Date: June 11, 2026. Available balance: NGN 150,000.",
         0),
        ("Slack notification with GitHub links",
         "CHAOSS Community via Slack. Your team has sent 181 messages recently. 1 more teammate has joined. View all unread messages.",
         0),
        ("Classic phishing — account suspended",
         "URGENT: Your account has been suspended. Click here to verify within 24 hours or your account will be permanently closed. Enter your username password and card details.",
         1),
        ("Lottery scam",
         "CONGRATULATIONS You have won 1500000 dollars in our international lottery. To claim your prize send your bank account details and pay a processing fee of 250 dollars.",
         1),
        ("AI-style BEC",
         "Hi, I need your help with something urgent. I am in a meeting and need you to process a wire transfer of 45000 dollars to our vendor today. Please use the bank details below and keep this confidential for now.",
         2),
    ]

    for name, text, expected in tests:
        cleaned_t = clean_text(text)
        X_t = vec.transform([cleaned_t])
        lr_p  = lr.predict_proba(X_t)[0]
        svm_p = svm.predict_proba(X_t)[0]
        proba = (lr_p + svm_p) / 2
        pred  = int(np.argmax(proba))
        conf  = float(np.max(proba)) * 100
        status = "✓" if pred == expected else "✗"
        print(f"  {status} [{LABELS[pred]:25s} {conf:5.1f}%] — {name}")

    # Save
    print("\nSaving models...")
    joblib.dump(lr,  os.path.join(BASE, "model.pkl"))
    joblib.dump(vec, os.path.join(BASE, "vectorizer.pkl"))
    joblib.dump(svm, os.path.join(BASE, "svm_final.pkl"))

    # Save label info for LIME
    lime_components = {
        "class_names":   ["legitimate", "traditional phishing", "ai generated phishing"],
        "label_mapping": {0: "legitimate", 1: "traditional phishing", 2: "ai generated phishing"},
    }
    joblib.dump(lime_components, os.path.join(BASE, "lime_components.pkl"))

    print(f"\n✓ Saved model.pkl, vectorizer.pkl, svm_final.pkl")
    print(f"✓ Ensemble accuracy: {ens_acc:.1%}")
    print("="*60)


if __name__ == "__main__":
    train()
