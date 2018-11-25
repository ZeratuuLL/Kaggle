What can be useful?
  * hour, minute, total length (use average time to estimate the last page), total length is time (Lifeng)
    * This is an example
    * Of how to write down findings
  * screenDepth in appInfo. Find the max one. (Lifeng)
  * Is Promotion empty for all? (Lifeng)
  
    When 'promotion' is not empty, it's a list of dictionaries, like product. Each dictionary has 4 keys:
    * promoCreative: some .jpg
    * promoID: promoName + promoPosition
    * promoName: Name
    * promoPosition: position
    
    Didn't find any relationship with other variables. Nothing extracted.
  * eCommerceAction's value (Lifeng)
  
    Many interesting findings:
  * Despite of many 0 in action_type, we still get a lot of meaningful actions
  * Only when action_type is '5', which means 'check out', the step can be different then 1. Need to look at more rolls to know whether this is true for 1.7M rows of data.
  * Strange, the numbers of (5,1), (5,2), (5,3), (6,1) are not decreasing. But if we only looking at rows instead of all pages, remove all duplicates in each row, then (5,1) (5,2) (5,3) will be decreasing. But (6,1) is still more than (5,3) but less than (5,2). Below is the result for all data. ```Counter({('0', '1'): 1705529, ('1', '1'): 250063, ('2', '1'): 244965, ('3', '1'): 88892, ('5', '1', 'Billing and Shipping'): 29618, ('5', '2', 'Payment'): 22963, ('6', '1'): 18559, ('4', '1'): 16707, ('5', '3', 'Review'): 15931, ('5', '1'): 84})```
  * 3 and 4 are paired with eventInfo
    
  * (customDimensions, customMetrics, customVariables) are they all empty lists? (Shitong)
  * how many things are there in contentGroup? contentGroupUniqueViews 1,2,3 or more?(Tongyi)
  * experiment (Lifeng)
    
    In the first 10000 lines, it's totally empty
    
  * What's in Item? (Lifeng)
     
    In the first 10000 lines, it's totally empty
    
  * In 'promotionActionInfo', most of them are 'promoIsView', some are 'promoIsClick'. More details? As well as eventInfo (Shitong)
  * publisher_infos, dataSource (Tongyi)
      
    In the first 10000 lines, publisher_infos is totally empty; dataSource takes 'web' or 'not set' (Some don't have this key).    
    
  * Are transactions and items paired? Anything interesting?(Shitong)
  * referer (Tongyi)

Try to find some insights. If there is anything valuable, think about what to extract. Write a function which can be directly applied to the dataset and save it in the dataprocessing py file. Add comments/docstrings to explain its function. Also write down your findings here
