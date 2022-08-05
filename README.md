# Credit-Card-Defaulters

## About

Tried to classify Credit-Card-Defaulters as Defaulters or NOT Defaulters using various ML algorithms. The notebook (Credit_card_defaulter.ipynb) consists of steps to process and explore the dataset, convert messages to vectors and applied ML techniques for the same.

## Dataset [link](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

There are 25 variables:

- `ID`: ID of each client
- `LIMIT_BAL`: Amount of given credit in NT dollars (includes individual and family/supplementary credit
- `SEX`: Gender (1=male, 2=female)
- `EDUCATION`: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
- `MARRIAGE`: Marital status (1=married, 2=single, 3=others)
- `AGE`: Age in years
- `PAY_0` to `PAY_6`: History of past payment. We tracked the past monthly payment records (from April to September, 2005) 

(-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … - 8=payment delay for eight months, 9=payment delay for nine months and above)

- `BILL_AMT1` to `BILL_AMT6`: Amount of bill statements.
- `PAY_AMT1` to `PAY_AMT6`: Amount of previous payments. 

Target Label: Whether a person shall default in the credit card payment or not.

- `default.payment.next.month`: Default payment (1=yes, 0=no)


## Exploratory Data Analysis


I looked at the distributions of the data and the value counts for the various categorical variables.

![alt text](https://github.com/hariranjanmeena/Credit-Card-Defaulters/blob/96a25da6a40fb51ae1bd0967be3d80c8aa1311f3/Images/eda1.PNG)

- From above plot we can infer that married people between age bracket of 30 and 50 and unmarried clients of age 20-30 tend to default payment with unmarried clients higher probability to default payment. Hence we can include MARRIAGE feature of clients to find probability of defaulting the payment next month

![alt text](https://github.com/hariranjanmeena/Credit-Card-Defaulters/blob/96a25da6a40fb51ae1bd0967be3d80c8aa1311f3/Images/eda2.PNG)

- It can be seen that females of age group 20-30 have very high tendency to default payment compared to males in all age brackets. Hence we can keep the SEX column of clients to predict probability of defaulting payment.


![alt text](https://github.com/hariranjanmeena/Credit-Card-Defaulters/blob/96a25da6a40fb51ae1bd0967be3d80c8aa1311f3/Images/eda3.PNG)

- Above plot indicates that there is higher proportion of clients for whom the bill amount is high but payment done against the same is very low. This we can infer since maximum number of datapoints are closely packed along the Y-axis near to 0 on X-axis

- No nan value present in dataset
- We can see that the dataset consists of 77% clients are not expected to default payment whereas 23% clients are expected to default the payment.
- SEX, EDUCATION, MARRIAGE, PAY_0, PAY_2, PAY_3,PAY_4, PAY_5, PAY_6and default payment next month are categorical columns
- In EDUCATION cloumn 4,5,6,0 represent same thing other/Unknown they could be combined as one.
- In MARRIAGE cloumn 3,0 represent same thing other they could be combined as one.
- PAY_0,PAY_2, PAY_3,PAY_4, PAY_5, PAY_6 contain non defined values like 0,-2 they should be fixed
- Marital status (3 = divorce; 0=others)
- PAY_AMTX: (-2 = No consumption; -1 = Paid in full; 0 = The use of revolving credit
- imbalaced dataset, 1(Yes) is more in taget column
- mostly all columns contain outliers
- gender ratio is 60% female and 40% male
- 20% of women's have default payment next month where as 24% of men's have default payment next month.
- 46% are university, 35% graduate school, 16% are high school students and rest other
- 26% of university have default payment next month
- 19% graduate school have default payment next month and 25.1% of high school have default payment next month
- 53.3% are single, 45% married and rest other
- 72.42% of people are of age group 20-40 whereas 26.67% belong to 41-60 group and rest 61-80 are 0.91%



## Model Building 
  
I tried three different models:

![alt text](https://github.com/hariranjanmeena/Credit-Card-Defaulters/blob/aee6d3d703db4cafc20379426f61672538b219e9/Images/models.PNG)


- Best Accuracy and Precision by ExtraTreesClassifier with RandomOverSampler
