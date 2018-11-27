What can be useful?
  * In total 1708337 rows, 7561613 pages (hits actually)
  * hour, minute, total length (use average time to estimate the last page), total length is time (Lifeng)
    * This is an example
    * Of how to write down findings
  * screenDepth in appInfo. (Lifeng)
    * Totally 0
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
    * Strange, the numbers of (5,1), (5,2), (5,3), (6,1) are not decreasing. But if we only looking at rows instead of all pages, remove all duplicates in each row, then (5,1) (5,2) (5,3) will be decreasing. But (6,1) is still more than (5,3) but less than (5,2). Below is the result for all data. 
    
    ```Counter({('0', '1'): 1705529, ('1', '1'): 250063, ('2', '1'): 244965, ('3', '1'): 88892, ('5', '1', 'Billing and Shipping'): 29618, ('5', '2', 'Payment'): 22963, ('6', '1'): 18559, ('4', '1'): 16707, ('5', '3', 'Review'): 15931, ('5', '1'): 84})```
    
    data has been extracted
    * 3 and 4 are paired with eventInfo
    
  * (customDimensions, customMetrics, customVariables) are they all empty lists? (Shitong)
  
     For first 10000 rows, all empty list. The code is updated to check the whole data set.
     
  * how many things are there in contentGroup? contentGroupUniqueViews 1,2,3 or more?(Tongyi)
  
    Here is the count for (key: value) pairs in all data 
    
    ```Counter({('contentGroup4', '(not set)'): 7561613, ('contentGroup5', '(not set)'): 7561613, ('contentGroup1', '(not set)'): 7366790, ('contentGroup3', '(not set)'): 7226048, ('previousContentGroup4', '(not set)'): 5854491, ('previousContentGroup5', '(not set)'): 5854491, ('previousContentGroup1', '(not set)'): 5646849, ('previousContentGroup3', '(not set)'): 5445981, ('contentGroup2', '(not set)'): 3138278, ('previousContentGroup2', 'Apparel'): 1806523, ('previousContentGroup1', '(entrance)'): 1707122, ('previousContentGroup2', '(entrance)'): 1707122, ('previousContentGroup3', '(entrance)'): 1707122, ('previousContentGroup4', '(entrance)'): 1707122, ('previousContentGroup5', '(entrance)'): 1707122, ('contentGroupUniqueViews2', '1'): 1651860, ('contentGroup2', 'Apparel'): 1555904, ('previousContentGroup2', '(not set)'): 1068503, ('contentGroup2', 'Brands'): 801315, ('previousContentGroup2', 'Brands'): 713190, ('previousContentGroup2', 'Bags'): 616646, ('contentGroup2', 'Bags'): 585296, ('previousContentGroup2', 'Office'): 479746, ('previousContentGroup2', 'Accessories'): 434398, ('contentGroup2', 'Office'): 402230, ('contentGroup2', 'Accessories'): 398598, ('previousContentGroup2', 'Drinkware'): 338100, ('previousContentGroup2', 'Electronics'): 338040, ('contentGroup2', 'Electronics'): 322374, ('contentGroup2', 'Drinkware'): 302577, ('previousContentGroup3', 'Mens'): 273774, ('contentGroup3', 'Mens'): 231245, ('previousContentGroup3', 'Womens'): 134736, ('previousContentGroup1', 'Google'): 127021, ('contentGroupUniqueViews3', '1'): 122620, ('contentGroup1', 'Google'): 109657, ('contentGroupUniqueViews1', '1'): 107227, ('contentGroup3', 'Womens'): 104320, ('contentGroup1', 'YouTube'): 77505, ('previousContentGroup1', 'YouTube'): 72681, ('previousContentGroup2', 'Lifestyle'): 41372, ('contentGroup2', 'Lifestyle'): 33867, ('contentGroup2', 'Nest'): 21174, ('previousContentGroup2', 'Nest'): 17973, ('previousContentGroup1', 'Android'): 7940, ('contentGroup1', 'Android'): 7661})```
    
    As for only keys, the count is:
    
    ```Counter({'contentGroup1': 7561613, 'contentGroup2': 7561613, 'contentGroup3': 7561613, 'contentGroup4': 7561613, 'contentGroup5': 7561613, 'previousContentGroup1': 7561613, 'previousContentGroup2': 7561613, 'previousContentGroup3': 7561613, 'previousContentGroup4': 7561613, 'previousContentGroup5': 7561613, 'contentGroupUniqueViews2': 1651860, 'contentGroupUniqueViews3': 122620, 'contentGroupUniqueViews1': 107227})```
    
  * experiment (Lifeng)
    
    Totally empty.
    
  * What's in Item? (Lifeng)
     
    Totally empty.
    
  * In 'promotionActionInfo', most of them are 'promoIsView', some are 'promoIsClick'. More details? As well as eventInfo (Shitong)
    * eventInfo (8416) (discard)
      * at least two keys: eventAction, eventCategory. some have one more key: eventLabel
      * eventAction: 'Add to Cart': 1381,'Onsite Click': 63,'Product Click': 1210,'Promotion Click': 424,'Quickview Click': 5152,'Remove    from Cart': 186
      * eventCategory: Enhanced Ecommerce and Contact us 
      * eventLabel: for Contact us eventCategory, Email/Phone. need further check for all data rows. (not that useful). for Enhanced Ecommerce & Quickview Click, specific product name (not useful) no label information except for Onsite click and Quickview Click
     * promotion (7350) (discard)
       * promoCreative: product name + .jpg or .png
       * promoId: almost same product name + Row + some number
       * promoName: the product name part of promoId, but seems all Android are spelled as Andriod in promoName...
       * promoPosition: Row+some number part of the promoId
       
     
      
  * publisher_infos, dataSource (Tongyi)
    * publisher_infos is totally empty
    * dataSource only has two values. The count is ```Counter({'web': 4377060, '(not set)': 8767})```. We can look into single rows now. Like if someone has multiple clicks, do they have same value for this?
    
  * Are transactions and items paired? Anything interesting?(Shitong)
  * referer (Tongyi)

Try to find some insights. If there is anything valuable, think about what to extract. Write a function which can be directly applied to the dataset and save it in the dataprocessing py file. Add comments/docstrings to explain its function. Also write down your findings here
