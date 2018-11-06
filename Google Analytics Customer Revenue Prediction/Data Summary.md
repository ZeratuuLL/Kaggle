# Shitong:
# [A link to detailed data description](https://support.google.com/analytics/answer/3437719?hl=en)

Some things...
* 'transactionRevenue': the unit for is US dollar $* 10^6$. Other than the interger values, other values are NaN. Guess it means no value passed, which means no transaction.
* 'trafficSource.keyword': some of them are '(not provided)', some of them are NaN. Others are not clear. Details should be refered to data description. Refer to the jupyter notebook for details of counts.
* 'socialEngagementType': only 'Not Socially Engaged', does not seem useful.
* 'totals.newVisits': NaN implies not first visit, 1 is first time visit. meaningful NaN!
* 'totals.bounces': NaN implies not bounced session, 1 is bounced session. meaningful NaN! But what is a bounced session?
* 'totals.pageviews': should be integer, no idea what NaN means here.
* sessionId is fullvisitorId_visitId. less than number of rows. Look at 3 nonunique cases: two rows with same sessionId. Columns that are different: date(3/3), visitstartime(3/3), geoNetwork.networkDomain(2/3), total.bounces(2/3), total.hits(3/3), total.pageviews(3/3). For date, (3/3) consecutive days, but the order is not same as the row index order. For visitorstartime, (3/3) visits happen in North America/US, the visitstartimes are after 11pm and at the very beginning of 12am the next day in California time.
* visitStartTime: got a single line of code converting to human readable time.
* geoNetwork.country: there are '(not set)'. 1468.
* geoNetwork.city: 56% is 'not available in demo dataset'. 3.8% is '(not set)'. Similar rankings for counts of visits considering all and only buy visits. Mean revenue: 'Fort Collins' is the highest, obviously higher than others. Followed by other cities with obvious differences. The ranking of mean revenue is very different from the ranking of counts of visits.
* geoNetwork.cityId: 1 level 'not available in demo dataset'. 
* geoNetwork.continent: America is highest in counts of visits (all and buy). For counts of buy visits, other continents are less comparable to America. Mean revenue: America is the highest. could be considered as a useful variable, or maybe use subContinent instead. 0.1% is '(not set)'.
* geoNetwork.subContinent: similar as geoNetwork.continent in counts of visits, but different for mean revenue. Easten African stands out as the second highest.
* geoNetwork.country: United States dominated in counts of visits. Completely different story for mean revenue, further referred to the plots. The different behaviors here might be interesting.
* geoNetwork.latitude: 1 level.
* geoNetwork.longitude: 1 level.
* geoNetwork.metro: about 80% are 'not available in demo dataset' or '(not set)'. 'not available in demo dataset' dominates in counts of visits (both all and buy). For mean revenue, valid levels dominate. (interesting)
* geoNetwork.networkDomain: 27% '(not set)'. 16% unknown.unknown. As a result, these two dominates in counts of visits. For mean revenue: other valid levels stand out.
* geoNetwork.networkLocation: 1 level.
* geoNetwork.region: 56% 'not available in demo dataset', 3% '(not set)'. 'not available in demo dataset'. For counsts of visits, 'not available in demo dataset' and 'Califorina' dominate. For mean revenue, many other valid levels exceed these two. 
* mean revenue is revenue per visit, not per user.



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
      
      
# Tongyi:
# [Some explanation](https://support.google.com/analytics/answer/1033173)
* Those start with 'trafficSource':
* adContent: The ad content of the traffic source. 98.8% Nan. 44 types, ome could be combined. 
* adwordsClickInfo.adNetworkType: Network type. Nan: 97.6%. 2 types: google search and search partners. Almost all Google search.
* adwordsClickInfo.criteriaParameters：all not available.
* adwordsClickInfo.gclId: The Google Click ID. Nan: 97.6%. Same ID ---> Same person. Same person ---> Different ID.
* adwordsClickInfo.isVideoAd: True if it is a Trueview video ad. Nan: 97.6%. Same as adNetwork Type. All False. 
* adwordsClickInfo.page: Page number in search results where the ad was shown. Nan: 97.6%. Same as adNetwork Type. 
* adwordsClickInfo.slot: Position of the Ad. Nan: 97.6%. Same as adNetwork Type. "top" and "rhs"
* campaign: The campaign value.??
* campaignCode: only 1 non-missing value?
* isTrueDirect: This field will also be true if 2 successive but distinct sessions have exactly the same campaign details. ??
* medium: The medium of the traffic source. 5 types + none (143026) + not set (120)
* keyword：The keyword of the traffic source, usually set when the trafficSource.medium is "organic" or "cpc". Nan (502929) + not provided (366363) + many (3659) types
* referralPath: If trafficSource.medium is "referral", then this is set to the path of the referrer.
* source: No missing value. 379 sources + (direct). like referral/search engine. could be combined (by key word or source type?). 
* Ps: 'cpc' all from google except one 'bing'. Those start with 'ad' almost all 'cpc'.
* Comments: Unknown for prediction. Predict features? Interact with other information? 
