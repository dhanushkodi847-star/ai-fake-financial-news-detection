"""
Dataset Generator — Creates 500+ labeled financial news articles
for training the AI Fake Financial News Detection model.
Covers deep Indian financial context (SEBI, RBI, NSE, BSE, etc.)
"""

import csv
import os
import random

REAL_TEMPLATES = [
    # RBI & Monetary Policy
    ("RBI maintains repo rate at {rate}%", "The Reserve Bank of India in its latest bi-monthly monetary policy review decided to maintain the repo rate at {rate} percent. The Monetary Policy Committee voted {vote} in favour of the decision. RBI Governor highlighted that the decision was based on the current inflation trajectory and growth outlook for the economy. The standing deposit facility rate remains at {sdf}% and the marginal standing facility rate at {msf}%."),
    ("RBI raises repo rate by {bps} basis points to {rate}%", "In a move to curb inflation, the Reserve Bank of India has raised the repo rate by {bps} basis points to {rate} percent. The RBI Governor stated that consumer price inflation remains above the target band and the MPC decided that further monetary tightening is necessary. GDP growth forecast for the fiscal year has been revised to {gdp}%."),
    ("RBI introduces new guidelines for digital lending", "The Reserve Bank of India has issued comprehensive guidelines for digital lending platforms operating in India. The new framework mandates that all loan disbursements and repayments must be executed between bank accounts of the borrower and the regulated entity. Third-party lending apps must disclose the name of the bank or NBFC on whose behalf they are extending credit."),
    ("RBI launches {initiative} for financial inclusion", "The Reserve Bank of India has launched a new initiative called {initiative} aimed at promoting financial inclusion in rural India. The programme targets to bring {target} million unbanked citizens into the formal banking system within the next two years. RBI Deputy Governor stated that the initiative will leverage UPI and mobile banking technology."),
    ("RBI imposes penalty of Rs {amount} crore on {bank}", "The Reserve Bank of India has imposed a monetary penalty of Rs {amount} crore on {bank} for non-compliance with certain directions related to KYC norms and customer service standards. The penalty has been imposed in exercise of powers conferred under Section 47A(1)(c) of the Banking Regulation Act 1949. The bank has been given 30 days to rectify the deficiencies."),
    # SEBI & Markets
    ("SEBI tightens IPO listing norms for better investor protection", "The Securities and Exchange Board of India has tightened norms for Initial Public Offerings to enhance investor protection. Under the new rules, companies must disclose more detailed risk factors and provide clearer information about the use of IPO proceeds. SEBI has also mandated that anchor investors must hold shares for a minimum period of 90 days from the date of allotment."),
    ("SEBI introduces T+1 settlement cycle for equity markets", "SEBI has successfully implemented the T+1 settlement cycle across all stocks listed on NSE and BSE. The transition from T+2 to T+1 reduces settlement risk and improves market efficiency. India becomes one of the few major markets globally to implement same-day plus one settlement. Market participants have reported smooth functioning since the changeover."),
    ("SEBI strengthens mutual fund regulations", "The Securities and Exchange Board of India has strengthened regulations governing the mutual fund industry. New rules require fund houses to invest a minimum of {pct}% of their net worth in their own schemes. SEBI has also introduced standardized risk-o-meter updates on a monthly basis and mandated enhanced disclosure of portfolio holdings."),
    ("SEBI approves framework for social stock exchange", "SEBI has approved the framework for the Social Stock Exchange in India, allowing non-profit organisations and social enterprises to raise funds from the capital market. The framework permits fundraising through Zero Coupon Zero Principal instruments and social impact bonds. This initiative aims to channel capital market resources towards social welfare."),
    # Stock Market Performance
    ("Sensex closes {points} points higher at {level}", "The BSE Sensex gained {points} points to close at {level} today, driven by broad-based buying across sectors. The Nifty 50 index also ended higher at {nifty} points, gaining {nifty_pct}%. Top gainers included {gainer1}, {gainer2}, and {gainer3}. FIIs were net buyers with purchases worth Rs {fii} crore while DIIs invested Rs {dii} crore."),
    ("Nifty 50 hits all-time high of {level}", "The NSE Nifty 50 index hit a fresh all-time high of {level} points in today's trading session. Banking and IT stocks led the rally with the Bank Nifty gaining {bank_pct}%. Market breadth was positive with {adv} stocks advancing against {dec} stocks declining on the NSE. India VIX declined {vix}% indicating lower market volatility expectations."),
    ("Indian stock markets decline {pct}% on global concerns", "Indian equity markets ended lower for the third consecutive session with the Sensex falling {pct}% to close at {level}. Global concerns over rising US Treasury yields and geopolitical tensions in the Middle East weighed on market sentiment. FIIs were net sellers to the tune of Rs {fii} crore. Defensive sectors like pharma and FMCG outperformed."),
    # Company Earnings & Results
    ("{company} reports Q{q} net profit of Rs {profit} crore", "{company} reported a net profit of Rs {profit} crore for Q{q} FY{fy}, representing a {growth}% {direction} year-on-year. Revenue from operations stood at Rs {revenue} crore, {rev_dir} {rev_growth}% YoY. The company's operating margin expanded by {margin_bps} basis points to {margin}%. The board recommended a dividend of Rs {dividend} per share."),
    ("{company} announces share buyback worth Rs {amount} crore", "{company} has announced a share buyback programme worth Rs {amount} crore at a maximum price of Rs {price} per share. The buyback represents {pct}% of the company's paid-up capital and free reserves. The buyback will be conducted through the tender offer route and is expected to be completed within {months} months from the date of board resolution."),
    ("{company} wins order worth Rs {amount} crore", "{company} has won a new order worth Rs {amount} crore from {client} for {project}. The order is expected to be executed over a period of {period} months. The company's total order book now stands at Rs {orderbook} crore, providing revenue visibility for the next {years} years. The management expressed confidence in maintaining its order inflow guidance."),
    # Banking & Finance
    ("{bank} raises FD interest rates by {bps} basis points", "{bank} has revised its fixed deposit interest rates upward by {bps} basis points across select tenures. The highest rate offered is now {rate}% for deposits of {tenure} days tenure. Senior citizens will receive an additional premium of {senior_bps} basis points. The revised rates are effective from {date}."),
    ("{bank} reports NPA reduction to {npa}%", "{bank} has reported improvement in asset quality with gross Non-Performing Assets declining to {npa}% from {prev_npa}% in the previous quarter. Provision coverage ratio improved to {pcr}%. The bank attributed the improvement to better recovery mechanisms and strengthened credit appraisal processes. Net NPA stood at {net_npa}%."),
    ("India's banking sector credit growth reaches {growth}%", "Credit growth in India's banking sector has reached {growth}% year-on-year, according to the latest RBI data. Retail loans grew by {retail}% while corporate credit expanded by {corp}%. Personal loans and home loans continued to be the major growth drivers. The banking sector's overall capital adequacy ratio remained healthy at {car}%."),
    # Economy & GDP
    ("India GDP grows {rate}% in Q{q} FY{fy}", "India's Gross Domestic Product grew by {rate} percent in the {qname} quarter of FY{fy}, according to data released by the National Statistical Office. The growth was primarily driven by strong performance in the {sector1} and {sector2} sectors. The agriculture sector grew by {agri}% while the services sector expanded by {services}%. The full-year GDP growth estimate has been maintained at {annual}%."),
    ("India's current account deficit narrows to {cad}% of GDP", "India's current account deficit narrowed to {cad}% of GDP in Q{q} compared to {prev}% in the same period last year. The improvement was driven by a reduction in the merchandise trade deficit to ${trade}B and strong growth in services exports at ${services}B. Remittances from overseas Indians remained robust at ${remit}B during the quarter."),
    ("India's foreign exchange reserves reach ${reserves} billion", "India's foreign exchange reserves have reached ${reserves} billion, according to the latest weekly data from the Reserve Bank of India. The reserves include ${fca} billion in foreign currency assets, ${gold} billion in gold reserves, and ${sdr} billion in SDRs. The current reserves provide import cover of approximately {months} months."),
    # IPO & Fundraising
    ("{company} IPO opens with {sub}x oversubscription", "The IPO of {company} has been oversubscribed {sub} times on day {day} of the issue. The QIB portion was subscribed {qib}x, NII portion {nii}x, and retail portion {retail}x. The company is raising Rs {amount} crore through the IPO in the price band of Rs {low}-{high} per share. Analysts expect strong listing gains based on GMP trends."),
    ("{company} raises Rs {amount} crore via QIP", "{company} has raised Rs {amount} crore through a Qualified Institutional Placement at a price of Rs {price} per share. The QIP saw participation from {investors} domestic and international institutional investors. The funds will be used for {purpose}. The allotment was completed on {date} and shares will be listed within {days} working days."),
    # Insurance & IRDAI
    ("IRDAI introduces new health insurance portability norms", "The Insurance Regulatory and Development Authority of India has introduced simplified health insurance portability guidelines. Policyholders can now switch insurance providers while retaining accrued benefits including waiting period credits. Insurers are mandated to process portability requests within {days} working days. Pre-existing disease coverage will carry forward seamlessly."),
    # Pharma & Healthcare
    ("{company} receives US FDA approval for generic {drug}", "{company} has received approval from the US Food and Drug Administration for its generic version of {drug} tablets. The approved product has an estimated annual market size of ${market} million in the US. The company plans to launch the product in the {timeline} and expects it to contribute meaningfully to US revenue. This is the company's {nth} FDA approval this fiscal year."),
    # Technology & IT
    ("{company} signs ${value} million deal with {client_type}", "{company} has signed a ${value} million, {years}-year deal with a {client_type} for {services}. The engagement includes {detail1} and {detail2}. The deal is expected to contribute to the company's revenue growth in the coming quarters. This marks one of the largest deals signed by the company in the {sector} vertical."),
    # UPI & Digital Payments
    ("UPI transactions cross {billion} billion monthly mark", "The National Payments Corporation of India reported that UPI transactions have crossed the {billion} billion monthly mark. Total transaction value exceeded Rs {value} lakh crore. PhonePe maintained its market leadership with {phonepay}% share followed by Google Pay at {gpay}%. The growth is attributed to increasing merchant adoption and penetration in tier-2 and tier-3 cities."),
    # Energy & Infrastructure
    ("India adds {gw} GW renewable energy capacity", "India has added {gw} GW of renewable energy capacity during FY{fy}, taking the total installed renewable capacity to {total} GW. Solar energy contributed {solar} GW while wind energy added {wind} GW. The government remains committed to achieving {target} GW renewable energy capacity by {year}. Private sector investments in green energy have increased by {growth}%."),
    ("NHAI awards highway contracts worth Rs {amount} crore", "The National Highways Authority of India has awarded new highway construction contracts worth Rs {amount} crore across {states} states. The contracts cover construction of {km} km of national highways under the Bharatmala Pariyojana. Projects include {project1} and {project2}. The awarded projects are expected to be completed within {years} years."),
    # Mutual Funds
    ("Mutual fund industry AUM crosses Rs {aum} lakh crore", "The Indian mutual fund industry's assets under management have crossed the Rs {aum} lakh crore milestone. SIP contributions reached a new monthly record of Rs {sip} crore. The number of SIP accounts now exceeds {accounts} crore. Equity funds saw net inflows of Rs {equity} crore while debt funds recorded inflows of Rs {debt} crore during the month."),
    # Real Estate
    ("India real estate market records {growth}% growth in Q{q}", "The Indian real estate sector recorded {growth}% year-on-year growth in residential unit sales during Q{q}. {city1}, {city2}, and {city3} were the top performing markets. Average property prices across top 7 cities increased by {price_growth}%. New project launches increased by {launch}% compared to the same quarter last year."),
    # Auto Sector
    ("{company} reports {growth}% growth in monthly vehicle sales", "{company} reported a {growth}% year-on-year growth in total vehicle sales for {month}. Domestic sales stood at {domestic} units while exports reached {exports} units. The {segment} segment led the growth with {seg_growth}% increase. The company maintained its market share at {share}% in the domestic passenger vehicle market."),
    # Telecom
    ("{telco} adds {subs} million subscribers in Q{q}", "{telco} added {subs} million subscribers during Q{q}, taking its total user base to {total} million. Average Revenue Per User (ARPU) increased to Rs {arpu} from Rs {prev_arpu} in the previous quarter. Data consumption per user per month reached {data} GB. The company's 5G network now covers {cities} cities across India."),
]

FAKE_TEMPLATES = [
    # Bank panic
    ("URGENT: {bank} to freeze all accounts {timeframe}", "URGENT WARNING: {bank} has announced that all customer accounts will be frozen {timeframe}. Customers must withdraw all their money immediately. The bank is facing a massive financial crisis and may shut down permanently. ATMs will stop working. This information is from a reliable insider source."),
    ("BREAKING: All banks will be closed for {days} days", "In a shocking development the government has ordered all banks in India to remain closed for {days} days starting {when}. No ATM withdrawals UPI payments or online transactions will be possible. Citizens are advised to stock up on cash immediately. The decision was taken in a secret emergency cabinet meeting."),
    ("All bank deposits above Rs {amount} to be seized by government", "The Indian government has secretly passed an ordinance to confiscate all bank deposits exceeding Rs {amount}. The seized money will fund a classified {project}. Banks have been instructed to not alert customers. This will take effect {when} and there is no appeal process. Share this message with everyone you know."),
    # Market crash scams
    ("Stock market to crash {pct}% {timeframe} says insider", "A {source} has predicted that the Indian stock market will crash by {pct} percent {timeframe} due to a secret policy that will destroy the economy. All investors should sell their shares immediately. BSE and NSE will reportedly shut down permanently. This is confirmed insider information."),
    ("BREAKING: Sensex to be permanently discontinued", "The government has decided to permanently shut down the BSE Sensex and Nifty 50 indices. All stock trading in India will become illegal from {when}. Existing investors will lose all their investments. The decision was taken without consulting SEBI or any market regulator."),
    # Too good to be true
    ("{bank} offering {rate}% interest on savings accounts", "{bank} has announced a revolutionary savings account scheme offering {rate}% annual interest rate. This is {times} times higher than any other bank in India. Customers need to deposit a minimum of Rs {min_deposit} to avail this scheme. The offer is valid for a limited period only. Rush to your nearest branch."),
    ("{entity} giving free Rs {amount} to every Indian citizen", "{entity} has announced that it will give Rs {amount} to every Indian citizen as a special gift. Citizens need to {action} to claim their money. The total cost of this scheme is Rs {total} crore. This offer expires {expiry}. forward this to all your contacts."),
    ("Government guarantees {return}% returns on all mutual funds", "The Indian government has announced that all mutual fund investments will now guarantee a minimum return of {return}% per year. Any losses will be directly compensated from government funds. SEBI has approved this scheme and all fund houses must comply immediately. This is a once in a lifetime opportunity."),
    ("Earn Rs {amount} daily by investing just Rs {invest}", "A secret investment scheme discovered by a {person} guarantees earnings of Rs {amount} daily with just a one-time investment of Rs {invest}. This government-approved scheme has already made {count} people millionaires. Click {link} to register before the opportunity closes forever."),
    # Fake regulatory actions
    ("{regulator} chairman arrested for Rs {amount} crore fraud", "BREAKING: {regulator} chairman has been arrested in connection with a Rs {amount} crore financial fraud involving manipulation of {what}. The entire board has resigned. All {affected} are frozen indefinitely. {consequence}. This information was leaked by an anonymous whistleblower."),
    ("{regulator} dissolved by government effective immediately", "The Government of India has dissolved {regulator} with immediate effect through a presidential ordinance. All regulatory functions will be transferred to {replacement}. Existing regulations are null and void. This unprecedented step was taken due to {reason}. No official confirmation has been issued yet."),
    # Currency manipulation
    ("Indian Rupee to be replaced by {replacement}", "The Reserve Bank of India has officially decided to replace the Indian Rupee with {replacement} as the countrys legal tender. All existing currency notes will become worthless from {date}. Citizens must {action} within {days} days. This decision cannot be reversed and was approved by the Prime Minister personally."),
    ("RBI printing Rs {amount} lakh crore causing hyperinflation", "Sources reveal that the RBI is secretly printing Rs {amount} lakh crore worth of new currency notes. This will cause prices of all goods to increase by {pct}% within weeks. The information was leaked by a senior RBI official. The government is hiding this from the public to avoid panic."),
    # Corporate acquisition hoaxes
    ("{company1} acquired by {company2} for Rs {amount}", "In an unprecedented deal {company2} has acquired {company1} for just Rs {amount}. All {company1} employees will be terminated immediately. Customers of {company1} will lose access to their accounts and services. The deal was signed secretly without regulatory approval."),
    # Fake policy changes
    ("Government to tax all {platform} messages at Rs {tax} each", "The Indian government has decided to impose a tax of Rs {tax} on every message sent via {platform}. This includes text messages images and videos. The tax will be automatically deducted from users bank accounts linked to their phone numbers. There is no way to opt out of this scheme."),
    ("All {investment} declared illegal effective immediately", "An emergency ordinance has been passed declaring all {investment} investments as illegal in India. Anyone holding {investment} will face {penalty}. Government agencies have begun raiding homes and offices. All {investment} must be surrendered within {days} hours or face arrest. This cannot be appealed."),
    # Fake insider tips
    ("{stock} share price guaranteed to reach Rs {target} this week", "Inside sources confirm that {stock} share price will reach Rs {target} within this week due to a secret {event} that has not been announced yet. Investors who buy now will make {returns}x returns. This is a guaranteed tip from company board members. Dont miss this opportunity."),
    ("Secret government scheme pays {return}% monthly returns", "A classified government scheme available only to selected citizens pays {return}% returns every month with zero risk. You need to register through {link} using your Aadhaar and bank details. Only {slots} slots remaining. The scheme is not advertised publicly to prevent oversubscription."),
    # Extreme fear mongering
    ("India economy to collapse in {months} months leaked report", "A leaked confidential {source} report reveals that the Indian economy will completely collapse within {months} months. GDP will become negative and unemployment will reach {pct}%. All banks are expected to fail. The government has been suppressing this information. Prepare for the worst economic crisis in history."),
    ("ALERT: Withdraw all money from {bank} immediately", "URGENT ALERT: {bank} is about to declare bankruptcy. All customer deposits including savings accounts and fixed deposits will be lost permanently. The {amount} crore cyber breach has compromised all account data. There is no deposit insurance protection. Act NOW before it is too late."),
    # Phishing and scam
    ("{bank} crediting Rs {amount} to every account holder", "{bank} is crediting Rs {amount} to every savings and current account holder as part of its {occasion} celebration. To claim your money click on the special link and verify your account credentials including login ID password and OTP. This offer expires in {hours} hours. Dont miss out."),
    ("Forward this to claim Rs {amount} from PM relief fund", "The Prime Ministers Office has announced that every citizen who forwards this message to {count} people will receive Rs {amount} from the PM relief fund. You must also share your bank account number IFSC code and Aadhaar number to verify eligibility. This scheme closes {date}."),
    # Absurd claims
    ("{company} announces all products free for {period}", "{company} has announced that all its products and services will be completely free for {period} starting tomorrow. This includes {products}. The company says this is a goodwill gesture. Simply register on {link} to avail. No terms and conditions apply. Verified by government sources."),
    ("Gold price to drop to Rs {price} per gram due to discovery", "Scientists have discovered a revolutionary method to create {what} in laboratories causing the price to crash to Rs {price} per gram. All gold shops are shutting down across India. RBI is selling all its gold reserves immediately. The jewellery industry will cease to exist within months."),
    ("{entity} to shut down all operations in India", "{entity} has announced the permanent shutdown of all its operations across India effective {date}. All {what} will lose access to services. The decision was made without consulting {regulator} or the government. There is no alternative arrangement and {consequence}. This is not a rumor."),
]

def fill_real_template(title_t, text_t, idx):
    r = random.Random(idx * 7 + 42)
    companies = ["Reliance Industries", "TCS", "Infosys", "HDFC Bank", "ICICI Bank", "Wipro", "HCL Tech", "Bharti Airtel", "ITC", "Bajaj Finance", "SBI", "Kotak Mahindra Bank", "Asian Paints", "Larsen & Toubro", "Titan Company", "Sun Pharma", "Dr Reddys Labs", "Cipla", "Axis Bank", "Maruti Suzuki", "Tata Steel", "Power Grid Corp", "NTPC", "Coal India", "Adani Enterprises", "Hindustan Unilever", "Nestle India", "Tech Mahindra", "JSW Steel", "Grasim Industries", "Zomato", "Paytm", "Swiggy", "IRCTC", "LIC"]
    banks = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank", "Bank of Baroda", "Punjab National Bank", "Canara Bank", "Union Bank", "IndusInd Bank", "IDBI Bank", "Yes Bank", "Federal Bank", "Bandhan Bank", "RBL Bank"]
    cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"]
    sectors = ["manufacturing", "services", "IT", "banking", "pharma", "real estate", "infrastructure", "agriculture", "automobile", "telecom"]
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    telcos = ["Bharti Airtel", "Reliance Jio", "Vodafone Idea", "BSNL"]
    initiatives = ["Jan Dhan 2.0", "DigiSakshi", "Grameen Vittiya Sashaktikaran", "UPI Shakti", "BharatPay Plus"]
    segments = ["SUV", "passenger vehicle", "electric vehicle", "commercial vehicle", "compact car", "luxury"]
    client_types = ["leading European bank", "Fortune 500 healthcare company", "US financial services firm", "global telecom operator", "major retail chain", "Asia-Pacific insurance leader"]
    services_list = ["cloud migration and digital transformation", "AI-driven analytics implementation", "core banking modernization", "enterprise resource planning", "cybersecurity infrastructure upgrade"]
    
    vals = {
        "rate": r.choice(["5.5", "6.0", "6.25", "6.5", "6.75", "7.0"]),
        "bps": r.choice(["25", "35", "50"]),
        "vote": r.choice(["4-2", "5-1", "unanimously"]),
        "sdf": r.choice(["6.0", "6.25", "6.5"]),
        "msf": r.choice(["6.75", "7.0", "7.25"]),
        "gdp": r.choice(["6.3", "6.5", "6.8", "7.0", "7.2"]),
        "initiative": r.choice(initiatives),
        "target": r.choice(["50", "75", "100", "150"]),
        "amount": str(r.randint(1, 50) * 1000),
        "bank": r.choice(banks),
        "company": r.choice(companies),
        "pct": r.choice(["1.2", "1.5", "1.8", "2.0", "2.5"]),
        "points": str(r.randint(200, 1200)),
        "level": str(r.randint(68000, 82000)),
        "nifty": str(r.randint(20000, 25000)),
        "nifty_pct": r.choice(["0.8", "1.2", "1.5", "1.8"]),
        "gainer1": r.choice(companies), "gainer2": r.choice(companies), "gainer3": r.choice(companies),
        "fii": str(r.randint(500, 8000)),
        "dii": str(r.randint(500, 5000)),
        "bank_pct": r.choice(["1.5", "2.0", "2.5", "3.0"]),
        "adv": str(r.randint(1200, 2000)),
        "dec": str(r.randint(500, 1200)),
        "vix": r.choice(["3", "5", "7", "10"]),
        "q": str(r.randint(1, 4)),
        "fy": r.choice(["2025", "2026"]),
        "profit": str(r.randint(500, 25000)),
        "growth": str(r.randint(5, 45)),
        "direction": r.choice(["growth", "increase"]),
        "revenue": str(r.randint(5000, 250000)),
        "rev_dir": r.choice(["up", "growing"]),
        "rev_growth": str(r.randint(5, 30)),
        "margin_bps": str(r.randint(20, 300)),
        "margin": r.choice(["18.5", "20.2", "22.8", "24.5", "28.0"]),
        "dividend": str(r.randint(2, 50)),
        "price": str(r.randint(100, 5000)),
        "months": str(r.randint(3, 12)),
        "client": r.choice(["a government agency", "a leading private corporation", "an international conglomerate"]),
        "project": r.choice(["infrastructure development", "IT modernization", "power plant construction", "metro rail project"]),
        "period": str(r.randint(12, 48)),
        "orderbook": str(r.randint(10000, 100000)),
        "years": str(r.randint(2, 5)),
        "tenure": str(r.choice(["180", "270", "365", "444", "555", "777"])),
        "senior_bps": str(r.choice([25, 50, 75])),
        "date": r.choice(["March 15", "April 1", "May 10", "June 1", "July 15"]),
        "npa": r.choice(["1.8", "2.1", "2.5", "3.0"]),
        "prev_npa": r.choice(["2.5", "2.8", "3.5", "4.0"]),
        "pcr": r.choice(["72", "76", "80", "85"]),
        "net_npa": r.choice(["0.5", "0.7", "0.9", "1.1"]),
        "retail": str(r.randint(12, 25)),
        "corp": str(r.randint(5, 18)),
        "car": r.choice(["14.5", "15.2", "16.0", "16.8"]),
        "qname": r.choice(["April-June", "July-September", "October-December", "January-March"]),
        "sector1": r.choice(sectors), "sector2": r.choice(sectors),
        "agri": r.choice(["1.5", "2.0", "3.5", "4.0"]),
        "services": r.choice(["7.0", "7.5", "8.0", "8.5"]),
        "annual": r.choice(["6.5", "7.0", "7.5"]),
        "cad": r.choice(["0.8", "1.0", "1.2", "1.5"]),
        "prev": r.choice(["2.0", "2.5", "3.0", "3.8"]),
        "trade": str(r.randint(50, 80)),
        "remit": str(r.randint(20, 35)),
        "reserves": str(r.randint(600, 720)),
        "fca": str(r.randint(500, 620)),
        "gold": str(r.randint(40, 60)),
        "sdr": str(r.randint(15, 25)),
        "sub": r.choice(["3.5", "5.2", "8.0", "12.5", "18.0"]),
        "day": str(r.randint(1, 3)),
        "qib": r.choice(["5.0", "8.2", "12.0"]),
        "nii": r.choice(["2.5", "4.0", "6.5"]),
        "low": str(r.randint(100, 800)),
        "high": str(r.randint(800, 2000)),
        "investors": str(r.randint(15, 50)),
        "purpose": r.choice(["business expansion", "debt reduction", "working capital", "acquisition funding"]),
        "days": str(r.choice([5, 7, 10, 15])),
        "drug": r.choice(["Revlimid", "Eliquis", "Ibrance", "Xarelto", "Jardiance"]),
        "market": str(r.randint(500, 5000)),
        "timeline": r.choice(["current quarter", "next quarter", "second half of the fiscal year"]),
        "nth": r.choice(["5th", "8th", "12th", "15th"]),
        "value": str(r.randint(100, 2000)),
        "client_type": r.choice(client_types),
        "detail1": r.choice(["cloud infrastructure migration", "AI analytics platform", "digital customer experience"]),
        "detail2": r.choice(["cybersecurity enhancement", "process automation", "data lake implementation"]),
        "sector": r.choice(["BFSI", "healthcare", "retail", "manufacturing", "telecom"]),
        "billion": r.choice(["12", "13", "14", "15"]),
        "phonepay": str(r.randint(45, 50)),
        "gpay": str(r.randint(30, 38)),
        "gw": r.choice(["15", "18", "22", "25"]),
        "total": str(r.randint(150, 200)),
        "solar": r.choice(["10", "12", "15"]),
        "wind": r.choice(["3", "5", "7"]),
        "year": r.choice(["2030", "2035"]),
        "states": str(r.randint(5, 15)),
        "km": str(r.randint(500, 3000)),
        "project1": r.choice(["Delhi-Mumbai Expressway extension", "Chennai-Bangalore corridor", "Eastern Peripheral Expressway"]),
        "project2": r.choice(["Amritsar-Jamnagar Economic Corridor", "Varanasi-Ranchi Highway", "Hyderabad ORR expansion"]),
        "aum": str(r.randint(50, 70)),
        "sip": str(r.randint(16000, 22000)),
        "accounts": r.choice(["7.5", "8", "8.5", "9"]),
        "equity": str(r.randint(10000, 25000)),
        "debt": str(r.randint(5000, 15000)),
        "city1": r.choice(cities), "city2": r.choice(cities), "city3": r.choice(cities),
        "price_growth": str(r.randint(5, 15)),
        "launch": str(r.randint(10, 30)),
        "month": r.choice(months),
        "domestic": str(r.randint(10000, 200000)),
        "exports": str(r.randint(2000, 30000)),
        "segment": r.choice(segments),
        "seg_growth": str(r.randint(5, 40)),
        "share": r.choice(["12", "15", "18", "22", "42"]),
        "telco": r.choice(telcos),
        "subs": r.choice(["2.5", "3.8", "5.0", "7.2"]),
        "arpu": str(r.randint(150, 220)),
        "prev_arpu": str(r.randint(130, 190)),
        "data": r.choice(["22", "25", "28", "32"]),
        "cities": str(r.randint(200, 700)),
    }
    
    try:
        title = title_t.format(**vals)
        text = text_t.format(**vals)
    except (KeyError, IndexError):
        title = title_t
        text = text_t
    return title, text

def fill_fake_template(title_t, text_t, idx):
    r = random.Random(idx * 13 + 99)
    banks = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "PNB", "Bank of Baroda", "Canara Bank", "Yes Bank", "Kotak Bank", "IDBI Bank"]
    regulators = ["SEBI", "RBI", "IRDAI", "PFRDA", "NPCI"]
    companies = ["Reliance", "TCS", "Infosys", "Wipro", "Tata Motors", "Paytm", "Zomato", "IRCTC", "LIC", "Adani Group"]
    platforms = ["WhatsApp", "Telegram", "Instagram", "Facebook", "YouTube"]
    investments = ["cryptocurrency", "gold", "mutual fund", "fixed deposit", "real estate", "stock market"]
    stocks = ["Reliance Industries", "TCS", "Infosys", "Adani Enterprises", "Paytm", "Zomato", "LIC"]
    entities = ["Amazon", "Google", "Mukesh Ambani", "Elon Musk", "Prime Minister", "Jeff Bezos"]
    replacements_currency = ["Bitcoin", "Ethereum", "Digital Dollar", "Chinese Yuan", "a new global cryptocurrency"]
    
    vals = {
        "bank": r.choice(banks),
        "timeframe": r.choice(["tomorrow", "next week", "from midnight tonight", "within 24 hours"]),
        "days": str(r.choice([7, 15, 30, 60, 90])),
        "when": r.choice(["tomorrow", "next Monday", "midnight tonight", "next month"]),
        "amount": r.choice(["5 lakh", "10 lakh", "1 crore", "2 lakh", "50000"]),
        "project": r.choice(["defense project", "space mission", "nuclear program", "debt repayment"]),
        "pct": str(r.choice([50, 70, 80, 90, 95, 99, 1000])),
        "source": r.choice(["top Wall Street analyst", "leaked government report", "anonymous SEBI insider", "classified RBI document"]),
        "rate": str(r.choice([15, 20, 25, 30, 50, 100])),
        "times": str(r.choice([3, 5, 10, 15, 20])),
        "min_deposit": str(r.choice([100, 500, 1000, 5000])),
        "entity": r.choice(entities),
        "action": r.choice(["send Aadhaar and bank details to a WhatsApp number", "forward this message to 20 friends", "click on the special link", "register on the website"]),
        "total": str(r.randint(10000, 500000)),
        "expiry": r.choice(["tonight", "in 24 hours", "this weekend", "by end of month"]),
        "return": str(r.choice([25, 50, 100, 200, 500])),
        "invest": str(r.choice([100, 500, 1000, 5000])),
        "person": r.choice(["retired IAS officer", "ex-RBI employee", "MIT professor", "NASA scientist"]),
        "count": r.choice(["500", "1000", "5000", "10000"]),
        "link": r.choice(["a secret website", "this Telegram link", "the registration portal"]),
        "regulator": r.choice(regulators),
        "what": r.choice(["stock prices of 200 companies", "mutual fund NAVs", "insurance policies", "bank interest rates"]),
        "affected": r.choice(["mutual fund investments", "stock market trades", "insurance policies", "bank transactions"]),
        "consequence": r.choice(["The entire financial system is expected to collapse", "All investments are now worthless", "Markets will remain closed for 6 months"]),
        "replacement": r.choice(replacements_currency),
        "date": r.choice(["next month", "January 1", "immediately", "within 90 days"]),
        "company1": r.choice(companies),
        "company2": r.choice(["Amazon", "Google", "Tesla", "Apple", "Microsoft"]),
        "platform": r.choice(platforms),
        "tax": str(r.choice([1, 2, 5, 10])),
        "investment": r.choice(investments),
        "penalty": r.choice(["10 years imprisonment", "Rs 50 lakh fine", "permanent asset seizure", "lifetime ban from banking"]),
        "stock": r.choice(stocks),
        "target": str(r.randint(5000, 100000)),
        "event": r.choice(["merger with Google", "deal with Apple", "government contract worth lakhs of crores", "revolutionary patent"]),
        "returns": str(r.choice([5, 10, 20, 30, 50, 100])),
        "months": str(r.choice([2, 3, 6, 9, 12])),
        "slots": str(r.choice([50, 100, 500, 1000])),
        "hours": str(r.choice([6, 12, 24, 48])),
        "occasion": r.choice(["anniversary", "Independence Day", "Diwali", "New Year"]),
        "products": r.choice(["phones laptops and tablets", "all streaming services", "insurance and banking", "flights and hotels"]),
        "price": str(r.choice([100, 200, 500, 1000])),
        "period": r.choice(["30 days", "3 months", "one year", "forever"]),
    }
    
    try:
        title = title_t.format(**vals)
        text = text_t.format(**vals)
    except (KeyError, IndexError):
        title = title_t
        text = text_t
    return title, text


def generate_dataset(output_path, target_count=550):
    """Generate expanded dataset with balanced real/fake articles"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rows = []
    real_count = target_count // 2
    fake_count = target_count - real_count
    
    # Generate REAL articles
    for i in range(real_count):
        tmpl_idx = i % len(REAL_TEMPLATES)
        title_t, text_t = REAL_TEMPLATES[tmpl_idx]
        title, text = fill_real_template(title_t, text_t, i)
        rows.append((title, text, 1))
    
    # Generate FAKE articles
    for i in range(fake_count):
        tmpl_idx = i % len(FAKE_TEMPLATES)
        title_t, text_t = FAKE_TEMPLATES[tmpl_idx]
        title, text = fill_fake_template(title_t, text_t, i)
        rows.append((title, text, 0))
    
    # Shuffle
    random.Random(42).shuffle(rows)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'text', 'label'])
        for title, text, label in rows:
            writer.writerow([title, text, label])
    
    real = sum(1 for r in rows if r[2] == 1)
    fake = sum(1 for r in rows if r[2] == 0)
    print(f"✅ Dataset generated: {len(rows)} articles ({real} real, {fake} fake)")
    print(f"   Saved to: {output_path}")
    return len(rows)


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(__file__), '..', 'data', 'financial_news_dataset.csv')
    generate_dataset(os.path.abspath(output), target_count=550)
