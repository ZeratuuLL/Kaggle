# Shitong:
# [A link to detailed data description](https://support.google.com/analytics/answer/3437719?hl=en)

Some things...
* Column 'transactionRevenue': the unit for is US dollar $* 10^6$. Other than the interger values, other values are NaN. Guess it means no value passed, which means no transaction.
* Column 'trafficSource.keyword': some of them are '(not provided)', some of them are NaN. Others are not clear. Details should be refered to data description. Refer to the jupyter notebook for details of counts.
* Column 'socialEngagementType': only 'Not Socially Engaged', does not seem useful.
* Column 'totals.newVisits': NaN implies not first visit, 1 is first time visit. meaningful NaN!
* Column 'totals.bounces': NaN implies not bounced session, 1 is bounced session. meaningful NaN! But what is a bounced session?
* Column 'totals.pageviews': should be integer, no idea what NaN means here.
* sessionId is fullvisitorId_visitId. less than number of rows. Look at 3 nonunique cases: two rows with same sessionId. Columns that are different: date(3/3), visitstartime(3/3), geoNetwork.networkDomain(2/3), total.bounces(2/3), total.hits(3/3), total.pageviews(3/3). For date, (3/3) consecutive days, but the order is not same as the row index order. For visitorstartime, (3/3) visits happen in North America/US, the visitstartimes are after 11pm and at the very beginning of 12am the next day in California time.
* visitStartTime: got a single line of code converting to human readable time.
* geoNetwork.country: there are '(not set)'. 1468.


# Lifeng:

Things about different variables:
  * channelGrouping: 8 levels
  * data: **To be deleted since everything is in visitStartTime**
  * fullvisitorID: 714k visitors.
  * sessionId: almost unique for everyone
  * SocialEngagementType: none engaged, delete?
  * visitID: I have a count and a histogram
  * visitNumber: Similar to visitID, but much longer tail. Need to have a closer look
  * visitStarttime: Split into year, month, day, hour, minute (**done**)
  * Those start with 'device': 
      * browser: 54 levels, a lot ones. Now I plan to use **Beta($\alpha, \beta$) distribution as a prior** and **combine the browsers** with only a few records.
      * Category: 3 levels. Browser seem to be highly correlated with Category. A tree classifier should be able to deal with this
      * isMobile: False/True is approximately 3/1. Almost perfectly math Category. After reading the file I decided to **ignore this**.
      * Everything else: All NA, to be deleted
  * Those start with 'geoNetwork':
      * city: More than half missing. 648 cities with minimum count 3
      * continent: 6 levels, with one named '(not set)'
      * country: 222 levels, not sure if there is '(not set)'
      * metro: around 700k missing (not NA). And 92 meaningful names
      * networkDomain: around 390k missing (not NA). Around 28k levels
      * region: around 536k missing (not NA). 374 meaningful levels.
      * subContinent: 22 meaningful levels.
      * Other: NAs
  * Those start with 'trafficSource':
      * adContent: 22 types, I think there is no missing values
      * adwordsClickInfo.adNetworkType: 21k non-missing values
      * adwordsClickInfo.gclId:
      * adwordsClickInfo.isVideoAd: False and NA. To be deleted
      * adwordsClickInfo.page: Some numbers, around 21k non-missing values
      * adwordsClickInfo.slot: 2 levels, 21k non-missing values
      * campaign: 865k missing.... Highly likely to be deleted
      * campaignCode: only 1 non-missing value?
      * isTrueDirect: 274k True. Nothing else
      * medium: Meaning? Around 144k missing values (not NA)
      * referralPath: 1475 different sources. Perhaps we will look into it
      * source: 380 sources. Should be able to extract something
      * Others: NAs
  * Those start with 'totals'
      * bounces: Seems to be a bad thing if the value is 1. Perhaps related to other variables. NAs filled with 0.
      * hits: interesting numbers. A high number should be good. A crosstable with hits and bounces has been obtained. When **bounces=1, hits=1, 2 or 3** and more than **99.1%** is 1. When **bounces=1, hits>1** with probability around **99.99%**. Perhaps using hits will be enough.
      * newVisits: 703k of 1s. Filled the rest with 0. Totally agrees with visitNumber. **To be deleted**
      * pageviews: Aroun 741k records equal to hits. Perhaps a log transformation.
      * transactionRevenue: The prediction, 11515 records?
      * visits: all 1s. To be deleted
