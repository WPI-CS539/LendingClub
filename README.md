# LendingClub
LendingClub is a US peer-to-peer lending company, the company operates an online lending platform that enables borrowers to obtain a loan, and investors to purchase notes backed by payments made on loans. Borrowers can take out loans from Lending Club worth up to as much as $40,000. Investors purchase notes, which are assets corresponding to fractions of these loans. Different notes correspond to different loans and borrowers. There are several pieces of information pertaining to the borrower which can serve as potential indicators to his/her timeliness in repaying the loan. One of the ways in which an investor can determine which borrowers are riskiest is through the question and answer section on their application, which provides the investor with information on the borrow and the purpose of the requested loan. The goal of our analysis is to use other data (mostly numerical) to create an empirical prediction as to how likely a borrower is to repay his/her loan on time.

Motivation
Our objective will be to study what factors would influence the company's decision to lend money to clients, and to predict lenders’ future repayment performance. More precisely, there is a column named “Loan Status” which describes each lender’s loan status, the attribute contains 10 classes as shown as below:
![title](https://github.com/WPI-CS539/LendingClub/tree/master/figure/labels.png)
https://github.com/WPI-CS539/LendingClub/blob/master/figure/labels.png
<img src="https://github.com/WPI-CS539/LendingClub/blob/master/figure/labels.png" alt="hi" class="inline"/>

We classify “Current”, “Fully Paid” and “Does not meet the credit policy. Status:Fully Paid” as good status, and others as bad status, so it becomes a binary label.


