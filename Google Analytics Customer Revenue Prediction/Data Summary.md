What we explored:

What we found:

Shitong:
# [A link to detailed data description](https://support.google.com/analytics/answer/3437719?hl=en)

Some things...
* Column 'transactionRevenue': the unit for is US dollar $* 10^6$. Other than the interger values, other values are NaN. Guess it means no value passed, which means no transaction.
* Column 'trafficSource.keyword': some of them are '(not provided)', some of them are NaN. Others are not clear. Details should be refered to data description. Refer to the jupyter notebook for details of counts.
* Column 'socialEngagementType': only 'Not Socially Engaged', does not seem useful.
* Column 'totals.newVisits': NaN implies not first visit, 1 is first time visit. meaningful NaN!
* Column 'totals.bounces': NaN implies not bounced session, 1 is bounced session. meaningful NaN! But what is a bounced session?
* Column 'totals.pageviews': should be integer, no idea what NaN means here.
