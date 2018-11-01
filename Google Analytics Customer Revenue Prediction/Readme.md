This repository is for codes and analysis of the Kaggle competition [Google Analytics Customer Revenue Prediction](https://www.kaggle.com/c/ga-customer-revenue-prediction). In this competition I am colleborating with [Shitong](https://github.com/Shitong-Wei) and [Tongyi](https://github.com/ttyi11).

Here is a quick brief for the files:

  * Getting Started: Let's get started!
  * Start with understanding the data.
  
# [A link to detailed data description](https://support.google.com/analytics/answer/3437719?hl=en)

Some things...
* Column 'transactionRevenue': the unit for is US dollar $* 10^6$. Other than the interger values, other values are NaN. Guess it means no value passed, which means no transaction.
* Column 'trafficSource.keyword': some of them are '(not provided)', some of them are NaN. Others are not clear. Details should be refered to data description. Refer to the jupyter notebook for details of counts.
* Column 'socialEngagementType': only 'Not Socially Engaged', does not seem useful.
* Column 'totals.newVisits': NaN implies not first visit, 1 is first time visit. meaningful NaN!
* Column 'totals.bounces': NaN implies not bounced session, 1 is bounced session. meaningful NaN! But what is a bounced session?
* Column 'totals.pageviews': should be integer, no idea what NaN means here.






